# Objektivat e arritura me Simulated Annealing

Rezultatet përdorin të njëjtin model, objektiv dhe normalizim si projekti.
Fitness më negativ është më i mirë. Nuk janë vlerësime nga portali ESA.

| Rasti | Objektivi | Fitness i verifikuar | Seed i fazës finale |
| --- | ---: | ---: | ---: |
| Medium | <= -1.71 | -1.7792980368828077 | 100 |
| Hard | <= -1.50 | -1.5020820939916717 | 600 |

Zgjidhjet finale janë në solutions/medium.json dhe solutions/hard.json.
Përmbajnë vektorin x, historikun real të arkivit, temperaturat dhe lëvizjet
e pranuara. solutions/summary.json përmbledh dy rezultatet.

## Verifikimi

```bash
python scripts/verify_solution.py solutions/medium.json
python scripts/verify_solution.py solutions/hard.json
```

Të dy vektorët u rivlerësuan edhe duke rindërtuar matricat e tranzicionit me
tolerancë 1e-12, kundrejt 1e-10 të konfigurimit standard. Fitness-i nuk ndryshoi.
Në tri matjet, Medium ka respektivisht 40, 39 dhe 40 satelitë brenda rrjetit;
Hard ka 40 në secilën matje. Përjashtimi i pikave jashtë rrjetit është pjesë
e funksionit ekzistues të vlerësimit.

## Përsëritja e fazës finale

Këto janë kërkime nga gjendje të ruajtura, jo rezultate nga zero me një
buxhet total prej vetëm 10,005 vlerësimesh. Gjendjet fillestare të fazës
finale janë ruajtur te solutions/starts për t'i bërë provat të përsëritshme.

| Rasti | Fitness në nisje | Prova SA | Temperatura fillestare | Temperatura finale |
| --- | ---: | ---: | ---: | ---: |
| Medium | -1.6722189173111244 | 10,000 | 0.003 | 0.00001 |
| Hard | -1.494943486020226 | 10,000 | 0.0005 | 0.00001 |

Çdo fazë vlerëson fillimisht 5 kandidatë: gjendjen e ruajtur dhe katër
mutacione pranë saj, me madhësi maksimale zhvendosjeje 0.08.
Pastaj kryen 10,000 prova SA: gjithsej 10,005 vlerësime për fazën finale.
Kërkimet që krijuan gjendjet fillestare kanë kosto shtesë dhe nuk duhen
anashkaluar në një krahasim të buxhetit nga zero.

```bash
python scripts/seek_targets.py --case medium --start solutions/starts/medium.json --target -1.71 --attempts 1 --steps 10000 --initial-temperature 0.003 --final-temperature 0.00001 --seed 100
python scripts/seek_targets.py --case hard --start solutions/starts/hard.json --target -1.50 --attempts 1 --steps 10000 --initial-temperature 0.0005 --final-temperature 0.00001 --seed 600
```

Komandat krijojnë folderë të rinj nën results/targets. Pas çdo prove ruajnë
attempt-NN.json, best.json dhe summary.json. Ndërprerja e një prove nuk e
fshin rezultatin e provës së fundit të përfunduar.
Nëse objektivi është arritur që në hyrje, nuk kryhet kërkim shtesë.
Pas shterimit të numrit të provave, rezultati ruhet edhe nëse objektivi
nuk është arritur. Nuk ka garanci se çdo seed e arrin një objektiv.

Të dy komandat u ekzekutuan përsëri nga kopjet në solutions/starts.
Vektori x, fitness-i dhe historia dolën identikë me zgjidhjet e publikuara.

Historiku i gjendjeve fillestare ruhet siç është matur. Disa fusha të
prejardhjes, si resumed_from ose source_artifact, tregojnë folderë lokalë
historikë nën results që nuk publikohen të gjithë. Fusha reproduction_start
e zgjidhjeve finale tregon kopjen e publikuar që mjafton për fazën e fundit.
Përsëritshmëria numerike kontrollohet në të njëjtin mjedis Python/modeli;
ndryshimet e platformës mund të ndikojnë pika shumë pranë kufijve të qelizave.
