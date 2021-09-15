import csv
text_list = []
with open('女.csv')as f:
    f_csv = csv.reader(f)
    ten_short_list = []
    count = 0
    for row in f_csv:

        ten_short_list.append(row[0])
        count = count +1
        if  count==10:
            text_list.append(ten_short_list)
            ten_short_list = []
            count = 0
    text_list.append(ten_short_list)

with open('nv.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerows(text_list)