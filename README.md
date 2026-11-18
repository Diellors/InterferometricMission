# Misioni interferometrik

Implementim i shkruar për këtë projekt: dinamika CR3BP, integrimi numerik,
propagimi linear i formacionit, numërimi i baselines dhe kërkim sipas satelitëve.
Nuk importohet ose ekzekutohet kod nga repository tjetër.

| Rasti | Orbita | Satelitë | Rrjeti | Variabla |
| --- | --- | --- | --- | --- |
| easy | DRO | 5 | 11 × 11 | 30 |
| medium | DRO | 40 | 21 × 21 | 240 |
| hard | HALO | 40 | 21 × 21 | 240 |

## Nisja në PyCharm ose terminal

Hap këtë folder si projekt dhe zgjidh një interpreter Python 3.11 ose më të ri.
Simulimi, optimizimi dhe testet përdorin vetëm bibliotekën standarde të Python-it.
Për grafiqe instalo opsionalisht Matplotlib:

```bash
python -m pip install -r requirements.txt
python main.py --case easy --population 20 --generations 10
python main.py --config configs/easy.json --plot
python main.py --case all --population 5 --generations 1
```

Kaggle dhe GPU nuk kërkohen. Implementimi aktual përdor CPU.
Në Kaggle klono repository-n dhe ekzekuto të njëjtat komanda;
për një kopje ekzistuese përdor git pull përpara ekzekutimit.
Nëse projekti është dataset, duhet përditësuar dataset-i.
Rezultatet shkruhen nën folderin e projektit, i cili duhet të jetë i shkrueshëm
(p.sh. /kaggle/working në Kaggle).

## Struktura

```text
mission/
  dynamics.py       # fusha CR3BP, Jacobiani, integrimi adaptiv RK4
  baselines.py      # rrjeti dhe diferencat e dallueshme
  problems.py       # tri rastet, propagimi, funksioni i objektivit
  optimizer.py      # kërkimi me gjendje orbitale të realizueshme
  sampling.py       # drejtimet e kërkimit dhe kufizimet në tri matjet
  search_state.py   # përditësimi inkremental i baselines
  plotting.py       # grafiqet e formacionit dhe baselines
configs/            # parametrat për tri rastet
tests/              # teste numerike dhe kombinatorike
scripts/            # verifikimi i rezultateve
docs/               # modeli matematikor dhe kufizimet
results/            # rezultate lokale, të përjashtuara nga Git
main.py             # pika e nisjes në PyCharm
solve.py            # pikë alternative nisjeje
```

## Modeli

Vektori përmban të gjitha dx, pastaj dy, dz, dvx, dvy dhe dvz.
Pozicionet kufizohen në [-1,1], shpejtësitë në [-10,10].
Pozicionet relative propagohen me matricën e tranzicionit të gjendjes të
modelit të linearizuar rreth orbitës së anijes kryesore.

Në t=0, T dhe 2T projektojmë formacionin në XY, XZ dhe YZ.
Për çdo plan llogarisim diferencat e dallueshme të koordinatave në rrjet.
Fill factor është numri i këtyre diferencave pjesëtuar me (2G-1)^2.
Numërohen të dy drejtimet dhe diferenca zero për një rrjet jo bosh.

Objektivi është negativi i shumës më të vogël të tri fill factors në një
matje. Rezultati më negativ është më i mirë. Në dy matjet e fundit lejohet
shtrirje 1.23 herë, jo 1.23 në fuqi të numrit të periudhave.
Përjashtohen pikat me cilëndo koordinatë jashtë kubit të hapur (-1,1).

Parametrat e orbitave dhe përkufizimi i rasteve janë të dhënat hyrëse të
sfidës ESA SpOC 3. Nuk pretendojmë se i kemi zbuluar këto orbita.
Kodi i implementimit është shkruar këtu; kjo nuk nënkupton që algoritmet
matematikore ose tema e sfidës janë shpikje të projektit.
Historiku Git përmban versionin e mëparshëm që përdorte kod të jashtëm.

## Testimi dhe rezultatet

```bash
python -m unittest discover -s tests -v
python scripts/verify_solution.py results/easy/seed-42/easy.json
python Vizualization.py results/easy/seed-42/easy.json
```

Testet mbulojnë baselines, kufijtë e rrjetit, Jacobianin numerik,
zgjidhjen analitike të një ODE-je, konstantën e Jacobi-t, matricën e
tranzicionit kundrejt trajektoreve të perturbura, si dhe riprodhueshmërinë
dhe përzgjedhjen e optimizuesit.

Rezultatet ruhen në results/<case>/seed-<seed>/<case>.json.
Rinisja me të njëjtin rast dhe seed mbishkruan rezultatin.
Parametrat CLI kanë përparësi mbi konfigurimin JSON.
JSON është format lokal, jo skedar i garantuar për dorëzim në portal.

Testet dhe ekzekutimet e shkurtra nuk provojnë optimum global ose përputhje
bit për bit me vlerësues të jashtëm. Formacioni përdor dinamikë relative
të linearizuar, jo integrim të plotë nonlinear për çdo satelit.

## Përmirësimi i cilësisë

Algoritmi i ri përdor më mirë rrjetin duke shmangur grumbullimin e formacionit
pranë qendrës. Funksioni i objektivit dhe normalizimi nuk janë ndryshuar.
Për Easy, kufiri matematikor është -0.142857142857, prandaj -1.5 nuk është i mundur.
Për krahasimin e matjeve shiko docs/quality_comparison.md.

Për të vazhduar nga një rezultat i ruajtur:

```bash
python main.py --case hard --population 20 --generations 1000 --seed 42 --resume results/hard/seed-42/hard.json
```

Ky është kërkim i ri që ruan kandidatin fillestar më të mirë, jo rikthim i
popullatës së mëparshme. Nëse rruga e daljes përputhet me hyrjen, rezultati
mbishkruhet vetëm pas përfundimit dhe rivlerësimit të kërkimit.
