import csv
import os

# Read in the file
with open('datafile.csv', 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('?', '0')

# Write the file out again
with open('datafile.csv', 'w') as file:
  file.write(filedata)

with open("datafile.csv","r") as source:
    rdr= csv.reader( source )
    with open("train_data_int.csv","a") as result:
        wtr= csv.writer( result )
        for r in rdr:
            wtr.writerow((r[1], r[2], r[3], r[4],r[5], r[6], r[7], r[8], r[9]))

f = csv.reader(open("datafile.csv"))
for row in f:
  print (row[-1])
  with open('train_labels.csv', 'a') as the_file:
      the_file.write(row[-1]+ "\n")

with open('train_labels.csv', 'r') as file:
  filedata = file.read()

# Replace the target string
  filedata = filedata.replace('2', '-1')
  filedata = filedata.replace('4', '+1')

# Write the file out again
with open('train_labels.csv', 'w') as file:
  file.write(filedata)

with open('train_data_int.csv') as infile, open('train_data.csv', 'w') as outfile:
    for line in infile:
        if not line.strip(): continue
        outfile.write(line)


