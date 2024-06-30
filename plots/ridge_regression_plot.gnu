set title 'Ridge Regression Model'
set xlabel 'Feature'
set ylabel 'Target'
set style data linespoints
set key outside
set xrange [*:*]
set yrange [*:*]
set mouse
plot '-' using 1:2 title 'Original Data' with points pt 7 ps 1.5 lc rgb 'black', \
'-' using 1:2 title 'ALGEBRAIC Prediction' with lines lw 2 lc rgb 'red', \
'-' using 1:2 title 'BATCH Prediction' with lines lw 2 lc rgb 'blue', \
'-' using 1:2 title 'MINIBATCH Prediction' with lines lw 2 lc rgb 'green', \
1.000000 320.000000
2.500000 510.000000
3.700000 710.000000
4.800000 930.000000
5.200000 1090.000000
2.200000 420.000000
6.400000 1340.000000
4.100000 870.000000
9.800000 1980.000000
3.300000 770.000000
7.600000 1540.000000
8.300000 1720.000000
6.000000 1210.000000
5.700000 1150.000000
8.100000 1680.000000
5.900000 10000.000000
e
1.000000 222.336988
2.500000 507.677282
3.700000 735.949518
4.800000 945.199067
5.200000 1021.289812
2.200000 450.609223
6.400000 1249.562048
4.100000 812.040263
9.800000 1896.333382
3.300000 659.858773
7.600000 1477.834283
8.300000 1610.993088
6.000000 1173.471303
5.700000 1116.403244
8.100000 1572.947715
5.900000 1154.448616
e
1.000000 222.337007
2.500000 507.677296
3.700000 735.949527
4.800000 945.199073
5.200000 1021.289816
2.200000 450.609238
6.400000 1249.562048
4.100000 812.040271
9.800000 1896.333370
3.300000 659.858783
7.600000 1477.834279
8.300000 1610.993081
6.000000 1173.471304
5.700000 1116.403246
8.100000 1572.947709
5.900000 1154.448618
e
1.000000 222.337007
2.500000 507.677296
3.700000 735.949527
4.800000 945.199073
5.200000 1021.289816
2.200000 450.609238
6.400000 1249.562048
4.100000 812.040271
9.800000 1896.333370
3.300000 659.858783
7.600000 1477.834279
8.300000 1610.993081
6.000000 1173.471304
5.700000 1116.403246
8.100000 1572.947709
5.900000 1154.448618
e
