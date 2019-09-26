# LHYP

A kiinduló forrás tartalmaz kódot a con filok beolvasására valamint a rövid tengely (sa) képek olvasására. Ki is lehet próbálni, van egy kis példa program hozzá, ami rárajzolja a kontúrt a képre.

Mindenki fork-olja ezt a repót, majd készítsen egy saját branch-et. Időnként, mikor találunk valami érdekeset, szinkronizáljuk majd az egyes repókat.

A nano szerveren elérhetők példa adatok, amik alapján az adatbeolvasás és később a modellhez az adatbetöltés megírható. Javasolt a modellhez szükséges adatok (minták) pickle fájlba írása. Egy pácienshez egy pickle, minden szükséges információval, ami szükséges róla. Rövid tengelynél csak a diasztole fázis kell (tehát a systole nem). Ezt úgy lehet megtalálni, hogy egy adott slice esetén meg kell keresni a legnagyobb területű kontúrt.
