import os
from moco_simulation import run_moco_simulation

def run_batch_simulations(count=10, output_folder="results"):
    # Erstelle den Ordner, falls er noch nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Ordner '{output_folder}' wurde erstellt.")

    print(f"Starte Batch-Lauf für {count} Simulationen...")

    for i in range(1, count + 1):
        # 1. Simulation ausführen und Objekt erhalten
        solution = run_moco_simulation()

        # 2. Dateinamen generieren (z.B. simulation1.sto)
        file_name = f"simulation{i}.sto"
        file_path = os.path.join(output_folder, file_name)

        # 3. Das Objekt als Datei schreiben
        try:
            solution.write(file_path)
            print(f"[{i}/{count}] Gespeichert: {file_name}")
        except Exception as e:
            print(f"Fehler beim Speichern von {file_name}: {e}")

    print("\nBatch-Lauf erfolgreich abgeschlossen.")

if __name__ == "__main__":
    run_batch_simulations(10)