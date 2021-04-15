import datetime

print ('Current date/time: {}'.format(datetime.datetime.now()))
name = "Rami"
name = name+'_'+format(datetime.datetime.now())
name = name.replace(":","_")
print(name)