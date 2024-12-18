# Objekterkennung & Förderbandsteuerung

## Projektbeschreibung
Dieses Projekt umfasst eine Anwendung zur Datensatzgenerierung und Objekterkennung auf einem Förderband. Es kombiniert eine GUI mit Kamera-Integration, motorisierte Förderbandsteuerung und Deep-Learning-gestützte Objekterkennung.

## Hauptfunktionen
- **Bilddatensammlung**: Sammlung von Bildern verschiedener Objektklassen mit einer Kamera.
- **Datensatz-Erstellung**: Umwandlung gesammelter Bilder in Feature-Vektoren.
- **Modelltraining**: Training eines Klassifikationsmodells mit einem Random Forest.
- **Live-Inferenz**: Objekterkennung in Echtzeit basierend auf dem trainierten Modell.
- **Förderbandsteuerung**: Ansteuerung eines Nema17-Motors zur Bewegung des Förderbands.

## Verzeichnisstruktur
```
📁 Projektverzeichnis
├── app.py                 # GUI zur Steuerung des gesamten Prozesses
├── collect_images.py      # Sammlung von Bilddaten
├── create_dataset.py      # Erstellung eines Datensatzes
├── inference_classifier.py # Inferenzmodul für die Objekterkennung
├── neuralnet_features.py  # Feature-Extraktion mit ResNet
├── train_classifier.py    # Training des Klassifikationsmodells
├── rpi_motor_control.py   # Motorsteuerung des Förderbands
├── requirements.txt       # Abhängigkeiten des Projekts
```

## Voraussetzungen
- Python 3.9 oder höher
- Hardware:
  - Raspberry Pi (optional für Motorsteuerung)
  - Kamera
  - Förderband mit Nema17-Motor

### Installierte Bibliotheken:
Die benötigten Bibliotheken sind in der Datei `requirements.txt` angegeben und können wie folgt installiert werden:

```bash
pip install -r requirements.txt
```

## Verwendung
### 1. Start der Anwendung:
Die GUI wird durch Ausführen von `app.py` gestartet:
```bash
python app.py
```

### 2. Ablauf:
1. Objektklassen in der GUI definieren.
2. Bilder für jede Klasse sammeln ("Collect Data").
3. Datensatz erstellen ("Create Dataset").
4. Modell trainieren ("Train Classifier").
5. Live-Inferenz durchführen ("Inference Classifier").

### 3. Förderbandsteuerung:
Die Bewegungsrichtung des Förderbands kann über die GUI gesteuert werden (vorwärts/rückwärts/automatisch).

## Technologieübersicht
- **GUI**: tkinter
- **Feature-Extraktion**: ResNet18 (vortrainiert)
- **Klassifikation**: Random Forest Classifier (scikit-learn)
- **Motorsteuerung**: RpiMotorLib (für Nema17-Motor)

