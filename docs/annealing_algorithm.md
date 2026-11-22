# Simulated Annealing për misionin interferometrik

SA është algoritëm kryesor i zgjedhshëm me `--algorithm annealing` dhe tani
zgjedhja standarde e main.py. Implementimi ynë përdor vetëm bibliotekën
standarde Python. Ai përshtat një teknikë të njohur kërkimi; nuk pretendohet
një algoritëm i ri shkencor ose optimum global i garantuar.

## Gjendja, fqinjësia dhe objektivi

Gjendja është vektori x me 6N komponentë. Një fqinj ndryshon vetëm një satelit:
lëvizje përgjatë një segmenti të realizueshëm, gjendje e re e realizueshme ose
eksplorim pranë kufijve. Kontrollet orbitale dhe rrjetet janë të pandryshuara.
Objektivi mbetet negativi i mbulimit në matjen më të dobët.

SA mban dy gjendje: `current` që mund të përkeqësohet dhe `best` që ruan
zgjidhjen më të mirë të vizituar. Pranimi përdor vetëm objektivin kryesor.
Barazimet e arkivit mund të zgjidhen nga shuma e tri matjeve, pa ndryshuar
fitness-in e raportuar.

## Rregulli i pranimit dhe ftohja

Për delta = f(candidate) - f(current):

```text
nëse delta <= 0: prano
përndryshe: prano kur U(0,1) < exp(-delta / T)
```

Me L prova, në provën i përdorim ftohje gjeometrike:

    T_i = exp(log(T_initial) + i/max(1,L-1) * (log(T_final)-log(T_initial)))

Temperaturat duhet të jenë pozitive dhe T_final <= T_initial.
Kur L=1 përdoret temperatura fillestare; kur L=0 nuk bëhet asnjë lëvizje.
Vlerat standarde janë 0.02 dhe 0.0001 dhe janë parametra eksperimentalë,
jo vlera të provuara si optimale për të tri rastet.

## Pseudokodi

```text
vlerëso P kandidatë fillestarë
current = kandidati fillestar më i mirë
best = current
për i = 0 deri L-1:
    llogarit temperaturën T_i
    candidate = lëviz një satelit të current
    vlerëso candidate
    delta = f(candidate) - f(current)
    prano sipas rregullit Metropolis
    nëse candidate është më i mirë se best: best = candidate
    në pikat e regjistrimit ruaj f(best), f(current), T_i
kthe best, jo gjendjen e fundit current
```

## Invariantet dhe kufizimet

Pas k provave, best është më i miri mes P kandidatëve fillestarë dhe k
kandidatëve të vlerësuar. Invarianti vlen në fillim dhe ruhet nga krahasimi
pas çdo prove. Prandaj historiku i best nuk rritet, edhe kur current përkeqësohet.
Ky është invariant i ruajtjes së më të mirës, jo provë e optimumit global.
Algoritmi përfundon pas numrit të fundëm L të provave.

Propozimet respektojnë kufizimet e orbitës dhe nuk pretendojmë simetri të
kernelit ose balancë të detajuar. Ky është përshtatim heuristik i SA; nuk
zbatojmë këtu ndonjë teoremë të konvergjencës globale për ftohje të pafundme.

## Kompleksiteti

N është numri i satelitëve, M numri i matjeve dhe G madhësia e rrjetit.
Ka tri plane për matje dhe dimensioni i gjendjes është konstant 6.
Le të jetë B=min(N^2,(2G-1)^2). Me operacione hash mesatare O(1):

- Inicializimi i P kandidatëve: O(P M N^2) për numërimet e baselines.
- Një provë ndryshon një satelit, por kopjon numërime dhe kontrollon vektorin:
  O(M(N+B)), jo O(1).
- Kërkimi: O(P M N^2 + L M(N+B)). Integrimi fillestar orbital është veçmas.
- Hapësira: O(PN + M(N+B) + H), ku H është numri i pikave të historikut.
  Arkivi, current dhe candidate janë të kufizuar në një numër konstant gjendjesh.

## Komandat dhe buxheti i krahasimit

```bash
python main.py --algorithm annealing --case easy --population 20 --generations 50 --seed 1
python main.py --algorithm orbit --case easy --population 20 --generations 50 --seed 1
```

Për SA, `population=P` është numri i kandidatëve fillestarë dhe i provave
për bllok regjistrimi; nuk ka P zinxhirë aktivë. `generations=G_b` është numri
i blloqeve. Kështu L=P*G_b dhe të dy metodat marrin P*(G_b+1) vlerësime.
Historia ka G_b+1 pika. Grafiku përdor etiketa checkpoint, jo pretendimin
se një bllok SA është një brez evolucionar.

Për secilin seed, të dy metodat krijojnë të njëjtën popullatë fillestare.
Vlerësimet e kontrollit final dhe ndërtimi i modelit nuk hyjnë në buxhetin
e kërkimit. Provat greedy kanë edhe vlerësime formacionesh të pjesshme dhe
nuk barazohen drejtpërdrejt me këtë buxhet.

```bash
python scripts/run_experiments.py --algorithm annealing --cases easy medium hard --seeds 1 2 3 --population 20 --generations 50
python scripts/run_experiments.py --algorithm orbit --cases easy medium hard --seeds 1 2 3 --population 20 --generations 50
```

Folderët e eksperimenteve janë të rinj për çdo ekzekutim. Tabelat CSV ruajnë
algoritmin dhe nuk përziejnë metodat në të njëjtën mesatare.
SA i main.py ruan te `results/annealing/<case>/seed-<seed>/<case>.json`.
Varianti orbit ruan rrugën ekzistuese `results/<case>/seed-<seed>/<case>.json`.

```bash
python scripts/verify_solution.py results/annealing/easy/seed-1/easy.json
python Vizualization.py results/annealing/easy/seed-1/easy.json --show
```

JSON ruan temperaturat, gjendjen aktuale në checkpoint, arkivin, numrin e
lëvizjeve të pranuara dhe numrin e përkeqësimeve të pranuara. Verifikuesi
kontrollon përputhshmërinë e këtyre fushave dhe rivlerëson zgjidhjen finale;
nuk provon autenticitetin e të gjithë historikut të ndërmjetëm.

## Prova e shkurtër e implementimit

U kryen dy seed (1 dhe 2), P=20 dhe 50 blloqe për çdo rast dhe metodë:
1,020 vlerësime për ekzekutim. Popullata fillestare dhe buxheti janë të njëjtë.
Të gjitha rezultatet u rivlerësuan me modelin e plotë të projektit.

| Rasti | Mesatarja SA | Mesatarja orbit | Më i miri SA | Më i miri orbit |
| --- | ---: | ---: | ---: | ---: |
| Easy | -0.142857143 | -0.142857143 | -0.142857143 | -0.142857143 |
| Medium | -1.528851874 | -1.511005354 | -1.530636526 | -1.562760262 |
| Hard | -1.252230815 | -1.189767995 | -1.265318263 | -1.214158239 |

Këto janë vetëm prova të shkurtra me dy seed. SA pati mesatare më të mirë në
Medium dhe Hard, por varianti orbit pati rezultatin individual më të mirë në
Medium. Nuk pretendojmë epërsi të përgjithshme ose përmirësim në çdo seed.
Këto matje nuk zëvendësojnë rekordet e kërkimeve më të gjata të dokumentuara
në quality_comparison.md dhe nuk janë verifikuar në portalin ESA.

Për ta përsëritur tabelën përdor komandat e eksperimenteve më sipër me
`--seeds 1 2`. Parametrat SA ishin T_initial=0.02 dhe T_final=0.0001.

## Baza teorike

Kirkpatrick, Gelatt dhe Vecchi, *Optimization by Simulated Annealing*, Science,
220(4598), 671–680 (1983): https://doi.org/10.1126/science.220.4598.671.
Përdorim idenë e njohur të pranimit dhe ftohjes, me implementim lokal të
përshtatur për formacionet orbitale. Ky referim nuk është importim kodi.
