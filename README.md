# Logistic Regression Projekt

Meine Umsetzung der Logistic Regression Übung aus dem Udemy-Kurs "Python für Data Science, Maschinelles Lernen & Visualization" im Rahmen der Angleichungsleistung.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/1-Logistic_Regression/main?filepath=Logistic_Regression_Solution.ipynb)

## Überblick
Vorhersage, ob ein Nutzer auf eine Online-Anzeige klickt, anhand von Merkmalen wie Alter, Einkommen, Internetnutzung und Zeit auf der Website. Modell: Logistic Regression (scikit-learn).

## Inhalt
* `Logistic_Regression_Solution.ipynb` - Haupt-Notebook mit Datenanalyse, Visualisierung und Modell
* `Advertising.csv` - Datensatz (1000 Einträge)

## Ausführung

1. Auf den **Binder-Badge** oben klicken, um das Notebook in myBinder zu starten.
2. Warten, bis die Umgebung geladen ist (kann 1-2 Minuten dauern).
3. `Logistic_Regression_Solution.ipynb` öffnen.
4. Alle Zellen nacheinander ausführen (*Run > Run All Cells*).
5. **Erwartete Ergebnisse:**
   - Histogramme und Jointplot (Alter vs. Einkommen)
   - Train/Test Split und Training des Modells
   - Classification Report mit Precision, Recall und F1-Score
   - Accuracy von ca. **0.91 - 0.97**

---

## Prüfungsaufgabe 2: Automatisierung und Testen

Ich habe das Projekt für Aufgabe 2 um Unit-Tests und Logging erweitert, nach dem Ansatz aus dem Artikel "Unit Testing and Logging for Data Science".

### Dateien
| Datei | Beschreibung |
|---|---|
| `model_logic.py` | Logistic Regression Logik mit `my_logger` und `my_timer` Dekoratoren |
| `test_model.py` | Unit-Tests für `predict()` (Accuracy) und `fit()` (Laufzeit) |
| `generate_test_data.py` | Skript zur Erzeugung der Testdaten |
| `train_data.csv` | Trainingsdaten (700 Zeilen) |
| `test_data.csv` | Testdaten (300 Zeilen) |
| `training.log` | Log-File mit Trainingsereignissen |

### Testfälle

**Testfall 1 - predict():** Das Modell wird auf `train_data.csv` trainiert und die Accuracy auf `test_data.csv` geprüft. Ziel: Accuracy > 0.85.

**Testfall 2 - fit():** Die Laufzeit der Trainingsfunktion wird gemessen und geprüft, ob sie unter 120% der Normzeit (0.5s) bleibt.

### Testergebnisse
```text
[Test predict()] Gemessene Accuracy: 0.9767
.
[Test fit()] Gemessene Dauer: 0.0191s (Limit: 0.6000s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.047s

OK
```

### Tests ausführen

1. Binder-Umgebung über den Badge oben starten.
2. **Terminal** öffnen (*File > New > Terminal*).
3. Folgenden Befehl ausführen:
   ```bash
   python -m unittest test_model -v
   ```
4. Die Tests laden die Daten aus `test_data.csv` und `train_data.csv`.
5. Beide Tests sollten mit `OK` durchlaufen.

Um die Testdaten neu zu generieren: `python generate_test_data.py`
