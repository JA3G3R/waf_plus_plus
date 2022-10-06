import csv 
fields = ['Method', 'Connection', 'Accept', 'Accept-Charset', 'Accept-Language', 'Cache-control', 'Pragma', 'User-Agent', 'Content-Type', 'POST-Data', 'Request', 'Class']

with open('post_test.csv','r') as get_csv:
    with open('post_requests','w') as wf:
        reader = csv.reader(get_csv)
        next(reader)
        s = ""
        for row in reader:
            
            s += row[-1]+" Request \n"
            s += "POST / HTTP/1.1\n"
            s += "Connection: "+row[2]+"\n"
            s += 'Accept: '+row[3]+"\n"
            s += "Accept-Charset: "+ row[4] + "\n"
            s += "Accept-Language: "+row[5] + "\n"
            s += "Cache-control: "+row[6]+"\n"
            s += "Pragma: "+row[7]+"\n"
            s += "User-Agent: "+ row[8] + "\n"
            s+= "Content-Type: "+ row[9] + "\n\n"
            s += row[10] + "\n\n"

        wf.write(s)
        

                
            