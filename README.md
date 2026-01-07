# Logistic Regression Projekt

Dieses Repository enthält ein Logistic Regression Projekt als Teil der Angleichungsleistungen im Modul "Data Science und Engineering mit Python".

## Projektüberblick
Das Ziel dieses Projekts ist es, vorherzusagen, ob ein Nutzer auf eine Anzeige geklickt hat, basierend auf verschiedenen Merkmalen.

## Inhalt
* `Logistic_Regression_Solution.ipynb`: Das Haupt-Notebook mit der Analyse und dem Modell.
* `Advertising.csv`: Der Datensatz, der für Training und Test verwendet wurde.

## Prüfungsaufgabe 2: Automatisierung und Testen

Dieses Projekt wurde gemäß den Anforderungen für Aufgabe 2 refaktoriert und mit automatisierten Tests sowie Logging ausgestattet.

### Struktur
- `model_logic.py`: Enthält die Kernlogik (Logistic Regression) sowie Logging-Funktionalität.
- `test_model.py`: Führt Unit-Tests zur Validierung der Modellgüte (Accuracy) und der Trainingslaufzeit durch.
- `training.log`: Protokolliert Trainingsereignisse.

### Testergebnisse
Die Tests wurden erfolgreich ausgeführt:
```text
[Test predict()] Gemessene Accuracy: 0.9767
.
[Test fit()] Gemessene Dauer: 0.0199s (Limit: 0.7500s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.048s

OK
```

## Nutzung
Das Notebook kann direkt über [myBinder](https://mybinder.org/v2/gh/Johannes-Steinle/1-Logistic_Regression/main?filepath=Logistic_Regression_Solution.ipynb) ausgeführt werden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/1-Logistic_Regression/main?filepath=Logistic_Regression_Solution.ipynb)
