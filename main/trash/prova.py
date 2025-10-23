from pathlib import Path

# percorso relativo al file stesso dello script
file_path = Path(__file__).parent / "prompts" / "prompt_1a.txt"

# leggi tutto il testo
full_text = file_path.read_text(encoding="utf-8")

print(full_text)