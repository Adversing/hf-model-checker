from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
import sys
import psutil
import torch
from huggingface_hub import HfApi
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console: Console = Console()

if not os.path.exists('quant_multipliers.json'):
    console.print("[red]Error:[/red] quant_multipliers.json not found!")
    console.print("Please ensure the file exists in the same directory as the script.")
    sys.exit(1)

with open('quant_multipliers.json', 'r') as file:
    QUANT_MULTIPLIERS: Dict[str, float] = json.load(file)

def get_system_memory() -> Tuple[float, float]:
    ram_gb: float = psutil.virtual_memory().total / (1024**3)
    vram_gb: float = 0.0
    if torch.cuda.is_available():
        device: int = torch.cuda.current_device()
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    return ram_gb, vram_gb

def get_best_quantization(available_ram_gb: float, available_vram_gb: float, 
                         model_size_gb: float, available_quants: List[str]) -> str:
    max_memory: float = max(available_ram_gb, available_vram_gb)
    filtered_quants: Dict[str, float] = {k: v for k, v in QUANT_MULTIPLIERS.items() 
                                       if k in available_quants}
    suitable_quants: List[str] = []
    for quant, multiplier in filtered_quants.items():
        required_memory: float = model_size_gb * (multiplier + 0.1)
        if required_memory <= max_memory:
            suitable_quants.append(quant)
    return suitable_quants[-1] if suitable_quants else "Model too large for available memory"

def estimate_ram_requirement(file_name: str, file_size_bytes: int) -> float:
    size_gb: float = file_size_bytes / (1024 ** 3)
    file_upper: str = file_name.upper()
    overhead_multiplier: float = 2.5
    
    for quant, mult in QUANT_MULTIPLIERS.items():
        if quant in file_upper:
            overhead_multiplier = mult
            break
    
    base_ram: float = size_gb * overhead_multiplier
    attention_overhead: float = size_gb * 0.1
    return base_ram + attention_overhead

def get_performance_label(estimated_ram_needed: float, user_ram_gb: float, 
                         vram_gb: float = 0) -> str:
    if vram_gb >= estimated_ram_needed:
        return f"[green]GPU-Ready[/green] (needs {estimated_ram_needed:.2f}GB, {vram_gb:.2f}GB VRAM available)"
    
    if estimated_ram_needed > user_ram_gb:
        return f"[red]Too large[/red] (needs {estimated_ram_needed:.2f}GB, {user_ram_gb:.2f}GB RAM available)"
    elif estimated_ram_needed > user_ram_gb * 0.5:
        return f"[yellow]Will be slow[/yellow] (needs {estimated_ram_needed:.2f}GB, {user_ram_gb:.2f}GB RAM available)"
    else:
        return f"[green]Ready[/green] (needs {estimated_ram_needed:.2f}GB, {user_ram_gb:.2f}GB RAM available)"

def group_split_files(files: List[Any]) -> Dict[str, Dict[str, Union[float, List[Any]]]]:
    grouped: Dict[str, Dict[str, Union[float, List[Any]]]] = {}
    for f in files:
        quant_type: Optional[str] = None
        filename_upper: str = f.rfilename.upper()
        
        for base_quant in QUANT_MULTIPLIERS.keys():
            if base_quant in filename_upper:
                size_variant: str = ""
                if "_L" in filename_upper:
                    size_variant = "_L"
                elif "_M" in filename_upper:
                    size_variant = "_M"
                elif "_S" in filename_upper:
                    size_variant = "_S"
                elif "_XS" in filename_upper:
                    size_variant = "_XS"
                elif "_XXS" in filename_upper:
                    size_variant = "_XXS"
                quant_type = f"{base_quant}{size_variant}"
                break
        
        if quant_type and quant_type not in grouped:
            grouped[quant_type] = {'size': 0.0, 'files': []}
        if quant_type:
            grouped[quant_type]['size'] += f.size
            grouped[quant_type]['files'].append(f)
    return grouped

def analyze_huggingface_url(repo_url: str) -> None:
    if "huggingface.co/" not in repo_url:
        console.print("[red]URL is not a valid Hugging Face model URL.[/red]")
        return

    repo_url = repo_url.rstrip('/')
    parts: List[str] = repo_url.split("huggingface.co/")
    path_part: str = parts[1]
    repo_id: str = (path_part.split("/tree/main/")[0] if "/tree/main/" in path_part 
                   else path_part.split("/blob/main/")[0] if "/blob/main/" in path_part 
                   else path_part)
    
    api: HfApi = HfApi()
    try:
        api.model_info(repo_id)
    except Exception:
        console.print("\n[yellow]Model Not Found[/yellow]")
        console.print("[red]The model you're looking for does not exist or it's not public.[/red]")
        return
    
    viable_quants: List[Tuple[str, float, str]] = []
    progress: Progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )
    
    with progress:
        progress.add_task("Analyzing system resources...", total=None)
        ram_gb, vram_gb = get_system_memory()
        
        if "/blob/main/" in path_part:
            repo_id, file_path = path_part.split("/blob/main/", 1)
            progress.add_task(f"Fetching info for {file_path}...", total=None)
            model_info = api.model_info(repo_id, files_metadata=True)
            all_files = model_info.siblings
            
            target_file = next((f for f in all_files if f.rfilename == file_path), None)
            if target_file:
                file_size_gb: float = target_file.size / (1024**3)
                estimated_ram_needed: float = estimate_ram_requirement(file_path, target_file.size)
                performance_label: str = get_performance_label(estimated_ram_needed, ram_gb, vram_gb)
                
                table: Table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("File", file_path)
                table.add_row("Size", f"{file_size_gb:.2f}GB")
                table.add_row("RAM Available", f"{ram_gb:.2f}GB")
                table.add_row("VRAM Available", f"{vram_gb:.2f}GB")
                table.add_row("Performance", performance_label)
                
                console.print(Panel(table, title="[bold]Model Analysis", border_style="green"))
                return

        subfolder: Optional[str] = None
        if "/tree/main/" in path_part:
            repo_id, subfolder = path_part.split("/tree/main/", 1)
        
        progress.add_task(f"Analyzing repository: {repo_id}...", total=None)
        model_info = api.model_info(repo_id, files_metadata=True)
        all_files = model_info.siblings
        
        if "GGUF" in repo_id.upper():
            gguf_files: List[Any] = [f for f in all_files if f.rfilename.endswith('.gguf')]
            if not gguf_files:
                console.print("[red]No GGUF files found in repository[/red]")
                return

            grouped_files: Dict[str, Dict[str, Union[float, List[Any]]]] = group_split_files(gguf_files)
            
            for quant_type, group in grouped_files.items():
                file_size_gb: float = float(group['size']) / (1024**3)
                estimated_ram_needed: float = estimate_ram_requirement(quant_type, float(group['size']))
                performance_label: str = get_performance_label(estimated_ram_needed, ram_gb, vram_gb)
                
                if "Too large" not in performance_label:
                    viable_quants.append((quant_type, file_size_gb, performance_label))

            viable_quants.sort(key=lambda x: x[1])

            table: Table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("RAM Available", f"{ram_gb:.2f}GB")
            table.add_row("VRAM Available", f"{vram_gb:.2f}GB")

        if viable_quants:
            gpu_ready_quants: List[Tuple[str, float, str]] = [(q, s, p) for q, s, p in viable_quants if "GPU-Ready" in p]
            ready_quants: List[Tuple[str, float, str]] = [(q, s, p) for q, s, p in viable_quants if "Ready" in p and "GPU-Ready" not in p]
            slow_quants: List[Tuple[str, float, str]] = [(q, s, p) for q, s, p in viable_quants if "Will be slow" in p]
            
            recommended: str
            if all("Will be slow" in p for _, _, p in viable_quants):
                recommended = min(viable_quants, key=lambda x: x[1])[0]
            else:
                if gpu_ready_quants:
                    recommended = max(gpu_ready_quants, key=lambda x: x[1])[0]
                elif ready_quants:
                    recommended = max(ready_quants, key=lambda x: x[1])[0]
                else:
                    recommended = min(slow_quants, key=lambda x: x[1])[0]
            
            quants_display: List[str] = []
            for quant, size, perf in viable_quants:
                if "GPU-Ready" in perf:
                    perf_display = f"[green]{perf.split('(')[0].strip()}[/green]"
                elif "slow" in perf:
                    perf_display = f"[yellow]{perf.split('(')[0].strip()}[/yellow]"
                else:
                    perf_display = f"[cyan]{perf.split('(')[0].strip()}[/cyan]"
                quants_display.append(f"{quant} ({size:.1f}GB) - {perf_display}")
            
            table.add_row("Viable Quantizations", "\n".join(quants_display))
            table.add_row("Recommended Quantization", f"[bold blue]{recommended}[/bold blue]")

            console.print(Panel(table, title="[bold]Model Analysis", border_style="green"))
        else:
            extensions: List[str] = [".safetensors", ".bin"]
            total_size_bytes: int = 0
            
            for f in all_files:
                if any(f.rfilename.endswith(ext) for ext in extensions):
                    if subfolder and not f.rfilename.startswith(subfolder):
                        continue
                    total_size_bytes += f.size
            
            if total_size_bytes == 0:
                console.print(f"[red]No model files ({', '.join(extensions)}) found in the specified repo/subfolder.[/red]")
                return
            
            estimated_ram_needed: float = estimate_ram_requirement(repo_id, total_size_bytes)
            performance_label: str = get_performance_label(estimated_ram_needed, ram_gb, vram_gb)
            total_size_gb: float = total_size_bytes / (1024 ** 3)
            
            table: Table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Model Size", f"{total_size_gb:.2f}GB")
            table.add_row("RAM Available", f"{ram_gb:.2f}GB")
            table.add_row("VRAM Available", f"{vram_gb:.2f}GB")
            table.add_row("Performance", performance_label)
            
            console.print(Panel(table, title="[bold]Model Analysis", border_style="green"))

if __name__ == "__main__":    
    console.print("\n[bold cyan]Enter a Hugging Face model URL (or 'exit' to quit):[/bold cyan]")
    url: str = input().strip()
        
    if url.lower() == 'exit':
        sys.exit()
            
    if url:
        console.print(f"\n[yellow]Analyzing:[/yellow] {url}")
        analyze_huggingface_url(url)
        console.print()
