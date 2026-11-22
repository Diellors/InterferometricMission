# Prova SA + kërkim lokal + lëvizje të dy satelitëve

Varianti në mission/hybrid.py kombinon eksplorimin SA dhe përmirësimin lokal.
Ai përdor të njëjtin objektiv dhe normalizim si modelet e tjera.

## Hapat

1. Vlerëso pesë kopje të së njëjtës gjendje fillestare, njësoj si kontrolli SA.
2. Shpenzo rreth 70% të provave të lëvizjeve në SA dhe ruaj arkivin më të mirë.
3. Rindërto numërimet e arkivit për fazën lokale; llogarite këtë si një vlerësim.
4. Shpenzo pjesën e mbetur në përmirësim lokal me madhësi hapash 0.005–0.16.
5. Kthe rezultatin më të mirë dhe rivlerësoje me modelin e plotë.

Faza SA mund të pranojë përkeqësime sipas rregullit Metropolis.
Faza lokale pranon vetëm përmirësime të rrepta të çelësit leksikografik:
negativi i matjes më të dobët, pastaj negativi i shumës së tri matjeve.
Historiku i rezultatit më të mirë nuk përkeqësohet.

## Lëvizja e dy satelitëve

Në 30% të propozimeve zgjidhen dy satelitë të ndryshëm. Për gjendje brenda
kufijve, njëri lëviz me +alpha*d dhe tjetri me -alpha*d. Intervali i alpha
kufizohet nga të dy satelitët në të tria matjet dhe nga kufijtë e variablave.
Kjo ruan mesataren e gjendjeve të çiftit. Në faqet e kufizimeve përdoret
një fallback me dy lëvizje të pavarura të realizueshme, pa këtë garanci për
mesataren. Pjesa tjetër e propozimeve përdor fqinjësinë ekzistuese me një satelit.

Vlerësuesi inkremental përditëson numërimet dy herë për çiftin, pa ndryshuar
prindin. Testet i krahasojnë këto numërime me vlerësuesin e plotë.

## Buxheti dhe përfundimi

Për buxhet E, zgjedhim A=floor(0.7*(E-6)) prova SA dhe L=E-6-A prova lokale.
Numri i vlerësimeve të kërkimit është saktësisht 5+A+1+L=E.
Kontrollet finale dhe raportimi trajtohen jashtë këtij buxheti.
Numri i fundëm i provave garanton përfundimin; nuk garanton optimum global.

Për M matje, N satelitë dhe B=min(N^2,(2G-1)^2) baselines për plan,
me hashing mesatar O(1), një përditësim i një ose dy satelitëve kushton
O(M(N+B)); dy satelitët shtojnë një faktor konstant. Kostoja totale e
kërkimit është O(MN^2 + E*M*(N+B)), pa integrimin orbital fillestar.
Hapësira për gjendjet dhe historikun është O(M(N+B)+H), me H checkpoint-e.

## Krahasimi i kontrolluar

```bash
python scripts/compare_hybrid.py --start solutions/starts/hard_1_502082.json --seeds 701 702 703 --evaluations 10005
```

Të dy metodat nisen nga i njëjti rekord dhe kanë të njëjtin buxhet kandidatësh.
SA përdor 10,000 lëvizje dhe pesë vlerësime fillestare. Hibridi e ndan buxhetin
mes dy fazave. Ky krahasim nuk mat vetëm efektin e lëvizjes së çiftit: ndryshojnë
edhe ndarja e buxhetit dhe faza e përmirësimit lokal.

Rezultatet, përmbledhja dhe CSV-ja ruhen në një folder të ri nën
results/hybrid_comparison. Rezultati fillestar nuk mbishkruhet.
Temperaturat standarde të kësaj prove janë 0.0005 dhe 0.00001.
Në mbrojtje duhen raportuar të gjithë seed-et dhe rastet pa përmirësim,
jo vetëm rezultati më i mirë.

## Rezultati i kësaj prove

Gjendja fillestare ishte Hard -1.5020820939916717 dhe ruhet te
solutions/starts/hard_1_502082.json. solutions/hard.json përmban rekordin e ri.
Çdo rresht më poshtë përdori 10,005 vlerësime kandidatësh.

| Seed | SA | SA + lokal + çift |
| --- | ---: | ---: |
| 701 | -1.5020820939916717 | -1.5020820939916717 |
| 702 | -1.5020820939916717 | -1.5020820939916717 |
| 703 | -1.5056513979773944 | -1.5020820939916717 |

Vetëm SA me seed 703 solli rekord të ri, me përmirësim 0.0035693039857227493
në madhësinë e fitness-it. Ky rezultat u rivlerësua edhe me tolerancë integrimi
1e-12 dhe nuk ndryshoi. Hibridi nuk përmirësoi rekordin në këto tri prova.
Prandaj kjo provë nuk mbështet pretendimin që hibridi është më i mirë.
Tri seed me një konfigurim nuk mjaftojnë për përfundime të përgjithshme.

Të dhënat e të gjitha provave janë në hybrid_comparison.csv pranë këtij dokumenti.
Koha varet nga mjedisi dhe implementimi i cache-it; krahasimi i cilësisë këtu
bazohet te i njëjti buxhet vlerësimesh, jo te pretendime për shpejtësinë.
