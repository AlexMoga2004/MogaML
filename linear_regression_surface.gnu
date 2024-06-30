set title 'Linear Regression Loss Surface'
set xlabel 'Weight (w)'
set ylabel 'Bias (b)'
set zlabel 'Loss'
set dgrid3d 100,100
set hidden3d
splot [250.000000:350.000000][-10.000000:100.000000] ('test/test_data/linear_regression_data.csv' using 1:2) \
 '+' \
(b + w * 1.000000 - 320.000000)**2 '+' \
(b + w * 2.500000 - 510.000000)**2 '+' \
(b + w * 3.700000 - 710.000000)**2 '+' \
(b + w * 4.800000 - 930.000000)**2 '+' \
(b + w * 5.200000 - 1090.000000)**2 '+' \
(b + w * 2.200000 - 420.000000)**2 '+' \
(b + w * 6.400000 - 1340.000000)**2 '+' \
(b + w * 4.100000 - 870.000000)**2 '+' \
(b + w * 9.800000 - 1980.000000)**2 '+' \
(b + w * 3.300000 - 770.000000)**2 '+' \
(b + w * 7.600000 - 1540.000000)**2 '+' \
(b + w * 8.300000 - 1720.000000)**2 '+' \
(b + w * 6.000000 - 1210.000000)**2 '+' \
(b + w * 5.700000 - 1150.000000)**2 '+' \
(b + w * 8.100000 - 1680.000000)**2 / 15

