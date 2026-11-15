# Misioni interferometrik — ESA SpOC 3

| Rasti | Orbita | Satelitë | Rrjeti | Variabla |
| --- | --- | --- | --- | --- |
| easy | DRO | 5 | 11 × 11 | 30 |
| medium | DRO | 40 | 21 × 21 | 240 |
| hard | HALO | 40 | 21 × 21 | 240 |

## Çfarë kërkojmë?

Për çdo satelit zgjedhim pozicionin dhe shpejtësinë relative fillestare.
Vektori ruhet sipas komponentëve: të gjitha dx, pastaj dy, dz, dvx, dvy, dvz.
Kufijtë zyrtarë janë [-1, 1] për pozicionet dhe [-10, 10] për shpejtësitë.
Kodi propagandon formacionin për t = 0, T, 2T dhe projekton pozicionet në XY, XZ, YZ.
Diferencat midis pikave japin baselines; sa më shumë diferenca të dallueshme,
aq më i madh fill factor.

KUJDES: README zyrtar shkruan -max, por të tri implementimet zyrtare
kthejnë `[-min(fill_factor)]`. Ky projekt përdor drejtpërdrejt implementimet:
minimizojmë negativin e shumës së fill factors në matjen më të dobët.
Një rezultat më negativ është më i mirë.
Kodi gjithashtu përdor faktorin 1.23 në të dyja matjet pasuese, pa e ngritur në fuqi.
Pikat përjashtohen nëse cilado koordinatë del jashtë intervalit të hapur (-1, 1),
pas aplikimit të këtij faktori.

## Instalimi dhe nisja

Nevojitet një instalim funksional Python me varësitë në requirements.txt.
Disponueshmëria e heyoka/pygmo varet nga platforma; nëse instalimi në Windows
nuk ofron paketa të përshtatshme, përdor Linux/WSL ose një mjedis conda të përshtatshëm.

```bash
conda env create -f environment.yml
conda activate interferometric-mission
python solve.py --case easy --generations 100 --plot
python solve.py --case medium --population 120 --generations 300 --plot
python solve.py --case hard --population 120 --generations 300 --plot
```

Për një provë të shkurtër të të tri rasteve:

```bash
python solve.py --case all --population 10 --generations 1
```

Algoritmi fillestar është Differential Evolution me inicializim që zvogëlon
shpërndarjen orbitale në tri matjet. Ai shërben si bazë krahasimi; rezultatet
nuk pretendojnë optimum global. Rritja e buxhetit dhe provat me disa seeds
janë hapat vijues për krahasim eksperimental.

Rezultatet ruhen në results/<case>/seed-<seed>/<case>.json; grafiqet
krijohen me --plot. JSON përmban vektorin x dhe fitness-in e rivlerësuar
me kodin zyrtar. Ky është format lokal rezultatesh, jo format dorëzimi në portal.
Për dorëzim përdor submission_helper të ESA-s sipas tutorialit zyrtar.

## Gjendja e verifikimit

Sintaksa e skedarëve Python dhe tri konfigurimet JSON u kontrolluan me sukses.
U gjet një Python i përfshirë në pgAdmin, por atij i mungojnë numpy, scipy,
matplotlib, heyoka dhe pygmo. Optimizimi nuk është ekzekutuar ende dhe nuk ka
rezultate numerike të verifikuara. Krijo mjedisin conda sipas udhëzimeve më sipër.

## Struktura e projektit

```text
mission/                 # ngarkimi i problemeve dhe algoritmi i optimizimit
configs/                 # parametrat për easy, medium dhe hard
scripts/                 # verifikimi i rezultateve me kodin zyrtar
docs/                    # shpjegimi i metodës dhe eksperimentet
notebooks/               # analiza interaktive
results/                 # rezultatet lokale dhe grafiqet (nuk ruhen në Git)
vendor/SpOC3-main/        # kodi, tutoriali, të dhënat dhe licencat e ESA-s
solve.py                 # pika e nisjes
environment.yml          # mjedisi conda
requirements.txt         # alternativë instalimi me pip
```

Përdor konfigurimet e gatshme dhe verifiko rezultatin:

```bash
python solve.py --config configs/easy.json --plot
python scripts/verify_solution.py results/easy/seed-42/easy.json
```

Parametrat e komandës kanë përparësi mbi konfigurimin JSON.
Rinisja me të njëjtin rast dhe seed mbishkruan rezultatin përkatës.
