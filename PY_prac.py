# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:44:00 2020

@author: DINESH_AGV
"""
#animal='Tiger'
#veg='Raddish'
#mineral='Iron'
#print("Here is an animal, a vegetable, and a mineral.")
#print(animal)
#print(veg)
#print(mineral)

#a=input("Please type something and press enter:")
#print('You entered:')
#print(a)
#
#text = input('What would you like the cat to say? ')
#text_length = len(text)
#print(' {}'.format('_' * text_length))
#print(' < {} >'.format(text))
#print(' {}'.format('-' * text_length))
#print(' /')
#print(' /\_/\ /')
#print('( o.o )')
#print(' > ^ <')

#print('Calculating the server cost')
#a=0.51*24 #per day cost
#b=a*30 #per month cost
#c=20 # no of days
#d=918 #Investment
#print('One server costs {} dollars per day' ' and {} dollars per a month'.format(a,b))
#print('Twenty servers costs {} dollars per day' ' and {} dollars per a month'.format(a*c,b*c))
#print('Number of days a server can be operated is {}'.format(d/a))
#print('Number of days twenty servers can be operated is {}'.format(d/(a*c)))

#miles=int(input('Number of miles you want to travel:'))
#if(miles<3):
#    print('You can go by walk or bicycle')
#elif(3 < miles < 300):
#    print('You can go by car')
#else:
#    print('You can go by AeroPlane')

#work_list=[]
#finished=False
#
#while not finished:
#    list=input('Enter a task for your to­do list. Press <enter> when done:')
#    if(len(list)==0):
#        finished=True
#    else:
#        work_list.append(list)
#        
#print()        
#print()
#print('YOUR TO_DO LIST')
#print('=='*8)
#for list in work_list:
#    print(list)

#dict={'Harsha':{'phone':'54454','email':'fdsfd'},
#      'AGV':{'phone':'45354543','email':'fsdfsdfd'}
#        
#      }
#
#
#for values in dict:
#    print(values)

dict1={'Jeff ':'Is afraid of clowns.',
'David':'Plays the piano.',
'Jason':'Can fly an airplane.'
}
for k,v in dict1.items():
    print(k,' : ',v)
    
print(dict1['Jason'])   
for k,v in dict1.items():
    print(k,' : ',v)