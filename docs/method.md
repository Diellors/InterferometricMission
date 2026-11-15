# Metoda dhe eksperimentet

1. Përdor funksionin zyrtar `udp.fitness` si referencë.
2. Inicializo popullatën duke zvogëluar drejtimet orbitale të paqëndrueshme me SVD.
3. Optimizo me Differential Evolution, pa gradient, sepse rrjeti krijon një objektiv me shkallë.
4. Rivlerëso zgjidhjen më të mirë dhe ruaj parametrat e eksperimentit.
5. Krahaso disa seeds me të njëjtin buxhet vlerësimesh, jo vetëm me të njëjtin numër brezash.

Përshkrimi i ESA-s ka `-max`, por kodi i shkarkuar ka `-min`.
Ky projekt ndjek kodin dhe vlerëson matjen më të dobët.
Shpejtësitë dhe pozicionet janë madhësi të normalizuara sipas modelit zyrtar.

Nuk kemi ende prova të ekzekutuara ose rezultate optimale të vërtetuara.
Për rastet me 240 variabla duhen eksperimente me buxhet më të madh dhe,
potencialisht, mutacione sipas satelitëve ose kërkim lokal mbi rrjetin.
