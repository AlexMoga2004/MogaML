set title 'KNN Classification'
set xlabel 'Feature 1'
set ylabel 'Feature 2'
set style data points
set terminal qt
set mouse
set pointsize 1.5
set palette defined (0 'red', 1 'green', 2 'blue', 3 'yellow')
plot 'train_data.tmp' using 1:2:3 with points palette title 'Training Data', \
     'new_data.tmp' using 1:2:($3) with points pt 7 ps 2 palette title 'New Points'
