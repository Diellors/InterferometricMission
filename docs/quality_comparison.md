# Krahasimi i cilësisë së optimizimit

Objektivi, orbitat, rrjetet dhe normalizimi janë të njëjtë për të dy algoritmet.
Fitness më negativ do të thotë rezultat më i mirë. Ky është vlerësim lokal i
modelit të projektit; rezultatet nuk janë verifikuar në portalin e sfidës.

## Krahasimi me buxhet të barabartë

Versioni bazë është optimizuesi nga commit `52d4f93`.
Versioni i ri është `feasible-satellite-search-v2` në këtë ndryshim.
Për secilin rast përdorëm seed-et 1, 2 dhe 3, popullatë 20 dhe 50 breza:
**1,020 vlerësime për çdo ekzekutim**. Seed-et janë të njëjta; popullatat
fillestare ndryshojnë sepse inicializimi është pjesë e përmirësimit.
Nuk po izolojmë vetëm efektin e një mutacioni.

| Rasti | Mesatarja bazë | Mesatarja e re | Më i miri i ri |
| --- | ---: | ---: | ---: |
| Easy | -0.095993953 | -0.142857143 | -0.142857143 |
| Medium | -0.219115606 | -1.500099147 | -1.562760262 |
| Hard | -0.215942891 | -1.187190165 | -1.214158239 |

Detajet për secilin seed ruhen në `quality_comparison.csv` pranë këtij dokumenti.
Për të përsëritur eksperimentin në secilin version:

```bash
python scripts/run_experiments.py --algorithm orbit --cases easy medium hard --seeds 1 2 3 --population 20 --generations 50
```

Koha nuk krahasohet këtu sepse disa matje u kryen ndërkohë që kishte procese
të tjera aktive. Numri i vlerësimeve është baza e krahasimit.

Tabela i përket variantit orbital v2. Nuk paraqet matje të Simulated Annealing.
Për SA duhen ekzekutime dhe krahasime të veçanta sipas docs/annealing_algorithm.md.

## Kërkimet më të gjata

Rekordet e mëvonshme me SA janë **Medium -1.7792980368828077** dhe
**Hard -1.5020820939916717**. Vektorët dhe gjendjet fillestare të fazës
finale janë publikuar nën solutions/. Shiko achieved_targets.md për
parametrat, rivlerësimin dhe kufijtë e krahasimit të buxhetit.
Këto nuk e ndryshojnë tabelën historike me 1,020 vlerësime më sipër.

Gjatë zhvillimit të variantit v2, para shtimit të drejtimeve që synojnë një
koordinatë të matjes, Medium arriti -1.672218917 me popullatë 20, 500 breza
dhe seed 1 (10,020 vlerësime). Hard arriti -1.436644854 me 1,000 breza,
popullatë 20 dhe seed 2 (20,020 vlerësime).

Një vazhdim nga ai rezultat Hard, me seed 42 dhe 3,000 breza të rinj,
arriti -1.466389054. Ky ekzekutim shtoi 60,020 vlerësime dhe ruajti rezultatin
te results/hard/seed-42/hard.json. Këto janë rezultate të kërkimeve më të gjata
gjatë zhvillimit, jo matje me të njëjtin buxhet si tabela më sipër.
Raportohen vetëm ekzekutimet e përfunduara dhe të rivlerësuara, jo vlerat
e regjistruara para ndërprerjes së një kërkimi.

Për Easy arritëm -1/7, kufirin e numrit të baselines, në të tre seed-et.
Për këtë rast, -1.5 është matematikisht e pamundur pa ndryshuar objektivin.
Për Medium dhe Hard nuk kemi provë të optimumit global ose garanci që çdo
seed dhe çdo buxhet do të përmirësojë rezultatin.
