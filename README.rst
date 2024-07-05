
==============================
 syndatagenerators README
==============================


Das ist die *README*-Datei für das Projekt *syndatagenerators*.

Diese Datei sollte UTF-8 kodiert sein und nutzt die Formatierungssprache
`reStructuredText <http://docutils.sourceforge.net/rst.html>`_.
Diese Readme wird potentiel genutzt um die Projekt-Webseite zu erstellen,
oder auch als Einstieg zu einem gitlab-Repository oder ähnlichem, sie richtet
sich also vornehmlich an Entwickler.

Hauptsächlich wird sie aber von Entwicklern direkt im Texteditor gelesen werden.
Sie ersetzt nicht die Dokumentation die sich im ``docs/`` Verzeichnis befindet.

Typischerweise enthält diese Datei eine Übersicht darüber wie eine Umgebung
für das Projekt einzurichten ist, einige Beispiele zur Nutzung der Build-Tools, usw.
Die Liste der Änderungen (Changelog) sollte nicht hier liegen (sondern in ``docs/changelog.rst``),
aber ein kurzer *“Aktuelle Änderungen”* Abschnitt für die neueste Version ist angemessen.


Repository
==========

Momentan ist als VCS für *syndatagenerators* GIT vorgesehen. Streng genommen könnte man aber
auch ein anderes Versionskontrollsystem (etwa SVN) verwenden.
In diesem Fall sollte man die``.gitignore`` löschen. In dieser Datei ist
aufgelistet welche Dateien nicht mit eingecheckt werden sollen. Das sind im
Allgemeinen alle automatisch generierten Dateien, also insbesondere
die generierte Dokumentation.

Im Ordner ``/tests`` wird die Struktur des Hauptordners
(z.B. Ordner ``syndatagenerators/syndatagenerators``) nachgebildet.
Alle Dateien im Ordner ``/tests`` haben das Prefix ``test_``,
also z.B. ``test_syndatagenerators.py``.

Notebooks die eingecheckt werden mit Namens- und Datums-Prefix versehen,
z.B. ``20180208_NKu_<Name>``.

**TODO:** Notebooks sollten überhaupt nicht in dieses Repository eingecheckt werden, außer
sie gehören wirklich zum Paket ... dann brauchen sie aber auch sicher kein Datums-Prefix.


Einrichtung & Konfiguration
===========================

**TODO:** Der ganze Abschnitt ist nicht besonders projektspezifisch und sollte im
*Confluence* untergebracht werden. Hier stände dann nur ein Link, bzw. besonderheiten von *Commons*.

Die Konfiguration des Projekts wird größtenteils über die ``setup.py`` vorgenommen.
Ein Beispiel für eine solche Datei findet sich im `Cookiecutter-Template <https://gitlab.cc-asp.fraunhofer.de/iee_oe224/Data_Science_project_template>`.
Üblicherweise müssen im oberen Bereich einige Einstellungen für das jeweilige Projekt angepasst werden.

Um das Modul in der Komandozeile verfügbar zu machen und in Python mittels ``import syndatagenerators``
importieren zu können, muss Python wissen wo es liegt. Eine Anleitung findet sich unter
http://confluence.iwes.fraunhofer.de/display/Organisation/OE224+Python.

Installation
------------
Die Installation wird über die Komandozeile durchgeführt.  
Das Repository ``clonen``::

    git clone https://gitlab.cc-asp.fraunhofer.de/sylaski/syndatagenerators.git

Gegebenenfalls ein Virtual Environment anlegen und starten: `doku <https://docs.python.org/3/library/venv.html>`_.

In den gecloneten Ordner syndatagenerators wechseln und die Requirements installieren (hier gezeigt mit pip, aber auch andere package 
manager möglich bspw. conda)::
    pip install -r requirements-dev.txt

Versioneer einrichten::

    versioneer install

Modul installieren entweder editierbar, d.h Änderungen im Code werden sofort durch einen Aufruf des Moduls zugänglich 
oder als Kompiliertes Packet, d.h. Änderungen werden erst zugänglich wenn das Modul neu installiert wurde. 

Editierbar bspw. mit pip::

    pip install -e .

Beachte den Punkt "." am Ende der Eingabe!

Nicht editierbar bspw. mit pip::

    pip install .

Beachte den Punkt "." am Ende der Eingabe!

Note: This Section needs revision/validation

Tests
=====

Tests werden mit *Unittest* ausgeführt.
Damit diese laufen muss das Modul ``syndatagenerators``
zum Import verfügbar sein (siehe oben).

Anschließend könnnen Tests über

    scripts\\test.bat

ausgeführt werden, wobei das aktuelle Verzeichnis das Projektverzeichnis sein muss.
(*Anmerkung* Der Backlash muss in RST-Dateien escaped werden, sodass hier im Quelltext zwei
 Schrägstriche stehen.)

Sofern *Coverage* installiert ist können die Tests auch inklusive
“Coverage-Report” ausgeführt werden:

    scripts\\test_with_coverage.bat

Dies erzeugt einen HTML-Report, der im (Standard-) Internet Browser angezeigt wird.
In diesem Report kann man direkt sehen welche Code-Teile noch nicht von Tests
abgedeckt sind. Das Ziel sind *100% test coverage*, wenn der Wert also darunter liegt
sollte man noch mehr Tests schreiben.


Dokumentation
=============

Die gesamte verfügbare Funktionalität sollte über Docstrings dokumentiert sein.
Das beinhaltet *Pakete*, *Module*, *Funktionen*, *Klassen* und *(public) Methoden*.
Nicht öffentliche Methoden und Funktionen sollten nach Möglichkeit auch dokumentiert sein
um die spätere Anpassung zu vereinfachen.

Wir verwenden `Sphinx <www.sphinx-doc.org/en/master/>`_ zur Dokumentationserstellung.
Dabei verwenden wir die Sprachanpassungen von
`Napoleon <sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_
mit `Google Style Docstrings <sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
für die Formattierung.
Docstrings müssen also entsprechend formatiert sein um später in der Doku richtig
dargestellt zu werden!

Um die Dokumentation zu erstellen ruft man

    scripts\\make_apidoc.bat

das neben der HTML-Dokumentation auch gleich die notwendigen *rst*-Dateien für die
Untermodule erstellt — alte Dateien werden ohne Nachfrage überschrieben!.
Die erstellten Module erkennt man daran dass ihr Name mit ``syndatagenerators.`` beginnt.
Die Warnung *„document isn't included in any toctree”* kann ignoriert werden — andere
Warnungen können dagegen auf Formatierungsprobleme hinweisen.

Die Standard *Sphinx* Dokumentationserstellung ohne *Sphinx Apidoc* ist über

    scripts\\make_doc.bat

zu erreichen.


Coding-Konventionen
===================

Bei Code-Anpassungen ist darauf zu achten, dass die Style-Konventionen eingehalten werden.

**TODO:** Link zu den Code-Konventionen im Confluence.

Daten
=====

London Smart Meter
------------------
Die LSM daten werden in einer PostgreSQL Datenbank auf ``applik-d208.iee.fraunhofer.de `` gehostet. Um auf die DB 
über ein Terminal zu zu greifen nutzt man den psql Client. In Ubuntu bspw.::
    sudo apt-get install postgresql

die DB erreicht man dann mit::
    psql -h <URL> -p <PORT> -d <DB_NAME> -U <USER_NAME> -W
bspw.::
    psql -h applik-d208.iee.fraunhofer.de -p 5432 -d last -U lastread -W

Dort können SQL Befehle ausgeführt werden, um mit der DB zu interagieren. ACHTUNG: Änderungen beeinflussen die Daten.

Sie hat die Structur::
                                            Table "public.cc_clc_fulldate_raw"
    Column  |            Type             | Collation | Nullable | Default | Storage  | Stats target | Description
    ----------+-----------------------------+-----------+----------+---------+----------+--------------+-------------
    level_0  | bigint                      |           |          |         | plain    |              |
    index    | bigint                      |           |          |         | plain    |              |
    LCLid    | text                        |           |          |         | extended |              |
    stdorToU | text                        |           |          |         | extended |              |
    DateTime | timestamp without time zone |           |          |         | plain    |              |
    kwh/hh   | double precision            |           |          |         | plain    |              |

A to B Datensätze
-----------------
AtoB Datensätze, die für das CycleGAN-Training benutzt werden können sind im Ordner ::
    syndatagenerators/data_sets/AtoB

zu finden. Die zugehörigen PyTorch Klassen sind ``TSToTSDataset`` (PyTorch ``Dataset``) und ``TSToTSDatamodule`` (PytorchLightning ``Datamodule``).
