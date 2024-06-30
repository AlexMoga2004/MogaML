# Define the coefficients
b = -0.2101346383
w1 = 0.6316231723
w2 = -1.5096572143

# Define the function for the decision boundary
f(x) = (1/w2) * (b + w1 * x)

# Gnuplot commands
set title 'Logistic Regression Model'
set xlabel 'Feature 1'
set ylabel 'Feature 2'
set style data points
set pointsize 1.5
set xrange [-5:5]
set yrange [-5:5]
set samples 1000

# Original Data
plot '-' using 1:2:($3 == 1 ? 1 : 2):($3 == 1 ? 2 : 1) with points pt variable lc variable title 'Original Data',\

0.838760 -2.307984 1
1.077187 -1.079137 1
-0.015668 0.915338 0
-0.563863 -2.719386 1
-0.224865 0.232743 0
-1.540645 1.087320 0
-1.125958 0.949409 0
0.347761 0.855021 0
0.050514 -0.448425 1
-0.059593 0.289930 0
-0.706220 0.422231 0
0.352585 -0.387367 1
1.310309 0.779300 0
-0.585351 -0.290425 1
-0.620955 -1.330311 1
-1.331212 -2.712353 1
-1.014249 0.591140 0
-0.515402 -0.663261 1
0.722886 0.239636 1
-1.058596 0.398750 0
-0.673781 1.369370 0
-1.192316 -0.027650 0
0.308618 0.963006 0
-0.296892 0.365172 1
-0.890134 0.150424 0
0.730009 1.302194 0
-0.547112 0.913336 0
-0.546090 -0.101325 0
1.062446 1.307896 0
0.226189 0.106427 1
0.472876 0.393864 0
2.810240 -2.545455 1
-0.697023 -0.571070 1
0.146342 0.407397 1
-0.457840 1.131232 0
2.123699 0.080098 1
-1.023302 -0.242975 0
0.681238 0.737006 0
-0.434947 -2.268630 1
-0.255905 0.446351 0
0.447755 -0.423996 1
-0.901188 -0.908057 1
-0.506508 1.091685 0
0.061717 -1.247603 1
-0.887449 1.506695 0
-0.522708 0.876296 0
-0.068207 -0.175856 1
0.463950 1.732770 0
0.292077 1.028646 0
0.330279 -1.188707 1
0.160091 -0.680238 1
0.027508 0.824989 0
-0.916748 -0.481050 1
0.852551 -1.019230 1
0.186222 -0.173717 1
0.073215 -0.458354 1
0.050840 0.591480 0
-1.021867 -1.689970 1
-1.279688 0.482173 0
0.937167 -0.925232 1
1.677481 -0.598289 1
-1.555492 0.841038 0
1.672300 -0.638523 1
1.346695 0.249952 1
-0.772768 -0.399532 1
-0.187846 -0.106971 1
1.130011 -0.292852 1
0.039044 -0.627480 1
0.058247 0.543813 0
-0.105107 -0.576493 1
0.309013 -1.633684 1
1.600750 1.046614 0
-0.107580 -1.319922 1
-0.839744 0.601400 0
0.117890 -0.501263 1
0.625167 -0.826819 1
-1.004336 -0.826031 1
-0.650576 -1.031928 1
-1.488283 0.234323 0
0.335435 -1.849399 1
1.606010 -0.100926 1
0.545830 1.789366 1
-0.124927 0.562787 0
-1.505603 -0.531690 0
0.771532 1.395740 0
0.306815 0.544681 0
-0.252999 -0.164822 1
1.535364 0.750171 0
-0.296304 -1.152193 1
1.136898 -0.290925 1
1.648388 0.344952 1
0.210819 1.323745 0
-1.676687 1.602347 0
-0.636200 0.431688 0
-0.178801 -0.698331 1
0.812351 -0.505851 1
-0.602100 -1.276719 1
0.129277 0.235941 0
-0.212306 -1.273111 1
0.526995 2.713513 1
e

# Predicted Data
plot '-' using 1:2:($3 > 0.5 ? 2 : 1) with points pt 7 lc variable title 'Predicted Data'

0.679296 0.934693 0.316074
0.383502 0.519416 0.417808
0.830965 0.034572 0.664363
0.053462 0.529700 0.364528
0.671149 0.007698 0.650770
0.383416 0.066842 0.586957
0.417486 0.686773 0.362862
0.588977 0.930436 0.305234
0.846167 0.526929 0.487281
0.091965 0.653919 0.327619
e

# Plot the decision boundary line
plot f(x) with lines title 'Decision Boundary'