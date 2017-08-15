import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.integrate import odeint
def rkck(yVector, currentTime, stepSize, Function):
    k1=stepSize*Function(currentTime,yVector)
    k2=stepSize*Function(currentTime+(1.0/5.0)*stepSize,yVector+(1.0/5.0)*k1)
    k3=stepSize*Function(currentTime+(3.0/10.0)*stepSize,yVector+(3.0/40.0)*k1+(9.0/40.0)*k2)
    k4=stepSize*Function(currentTime+(3.0/5.0)*stepSize,yVector+(3.0/10.0)*k1+(-9.0/10.0)*k2+(6.0/5.0)*k3)
    k5=stepSize*Function(currentTime+stepSize,yVector+(-11.0/54.0)*k1+(5.0/2.0)*k2+(-70.0/27.0)*k3+(35.0/27.0)*k4)
    k6=stepSize*Function(currentTime+(7.0/8.0)*stepSize, yVector+(1631.0/55296.0)*k1+(175.0/512.0)*k2+(575.0/13824.0)*k3+(44275.0/110592.0)*k4+(253.0/4096.0)*k5)

    return [k1,k2,k3,k4,k5,k6]

def errorEstimate(kVector):
    cVector=[37.0/378.0-2825.0/27648.0,
             0.0,
             250.0/621.0-18575.0/48384.0,
             125.0/594.0-13525.0/55296.0,
             0-277.0/14336.0,
             512.0/1771.0-1.0/4.0
    ]
    S=0
    for i in range(0,6):
        S=S+cVector[i]*kVector[i]
    return S


def solver_rkck_adaptive(InitialCondition,TimeRange,stepSize,Function):
    startTime,stopTime=TimeRange
    tryStepSize=stepSize
    tempStepSize=0
    # Initial User defined Step Size
    #currentStepSize=stepSize
    #newStepSize=currentStepSize
    # Initialize time
    t=startTime
    print("\n===================================================================")
    print("\tStarting Runge-Kutta Fourth and Fifth Order solver")
    print("\tInitial time starts at ", t)
    print("\tInitial step size guess is ", stepSize)
    print("\t------------------------------------------------------")
    print("\t|  ITER   |  NEXT STEP SIZE  | EXCEEDS BOUNDS? (Y/N) |")
    print("\t------------------------------------------------------")
    T=[]
    T.append(t)
    yVals=[]
    yVals.append(InitialCondition)
    k1,k2,k3,k4,k5,k6=rkck(yVals[-1],t,stepSize,Function)
    # Initial estimate for error
    #Delta0=errorEstimate([k1,k2,k3,k4,k5,k6])
    Delta0=1e-9
    SAFETY=0.9
    START=time.clock()
    COUNTER=0
    while t<stopTime:
        #print("h1=",newStepSize)
        tryStepSize=stepSize
        k1,k2,k3,k4,k5,k6=rkck(yVals[-1],t,tryStepSize,Function)
        Delta1=errorEstimate([k1,k2,k3,k4,k5,k6])
        #print(Delta1)
        if Delta0>=np.abs(Delta1):
            if np.abs(Delta1/Delta0)>1e-6:
                stepSize=(tryStepSize*SAFETY)*np.abs(Delta1/Delta0)**(-0.2)
            else:
                stepSize=tryStepSize*2.0
            print('\t|   '+'{:2d}'.format(COUNTER),'   |    '+'{:f}'.format(stepSize)+'      |           Y           |')
            # print("\tNEXT STEP SIZE WILL BE ", stepSize)
            yNext=yVals[-1]+(37.0/378.0)*k1+(250.0/621.0)*k3+(125.0/594.0)*k4+(512.0/1771.0)*k6
        elif np.abs(Delta1)>Delta0:
            newStepSize=(tryStepSize*SAFETY)*np.abs(Delta1/Delta0)**(-0.25)
            stepSize=max(newStepSize,0.1*tryStepSize) 
            k1,k2,k3,k4,k5,k6=rkck(yVals[-1],t,tryStepSize,Function)
            yNext=yVals[-1]+(37.0/378.0)*k1+(250.0/621.0)*k3+(125.0/594.0)*k4+(512.0/1771.0)*k6
            print('\t|   '+'{:2d}'.format(COUNTER),'   |    '+'{:f}'.format(stepSize)+'      |           N           |')
        t=t+stepSize
        yVals.append(yNext)
        T.append(t)
        COUNTER=COUNTER+1
    print("\t------------------------------------------------------")
    print("\tTIME TAKEN FOR PROCESS = ",time.clock()-START)
    print("===================================================================\n")
    return T,yVals


def solver_rkck_naive(InitialCondition,TimeRange,stepSize,Function):
    startTime,stopTime=TimeRange
    Time=np.arange(startTime+stepSize,stopTime,stepSize)
    #print(Time)
    yVals=[]
    yVals.append(InitialCondition)
    #print(yVals[-1])
    START=time.clock()
    for t in Time:
        [k1,k2,k3,k4,k4,k6]=rkck(yVals[-1],t,stepSize,Function)
        yNext=yVals[-1]+(37.0/378.0)*k1+(250.0/621.0)*k3+(125.0/594.0)*k4+(512.0/1771.0)*k6
        yVals.append(yNext)
    
    print("TIME TAKEN FOR NAIVE RK4/5 SOLVER = ",time.clock()-START)
    return yVals 

def rk4(yVector,currentTime,stepSize,Function):
    k1=Function(currentTime,yVector)
    k2=Function(currentTime+stepSize*0.5,yVector+0.5*k1*stepSize)
    k3=Function(currentTime+stepSize*0.5,yVector+0.5*k2*stepSize)
    k4=Function(currentTime+stepSize,yVector+k3*stepSize)
    yVectorNext=yVector+(k1+2.0*k2+2.0*k3+k4)*stepSize/6.0
    return yVectorNext


def solver_rk4(InitialCondition,TimeRange,stepSize,Function):
    startTime,stopTime=TimeRange
    Time=np.arange(startTime+stepSize,stopTime,stepSize)
    yVals=[InitialCondition]
    START=time.clock()
    for t in Time:
        yVals.append(rk4(yVals[-1],t,stepSize,Function))
    print("TIME TAKEN FOR RK4 SOLVER = ",time.clock()-START)
    return yVals
        
def somefun(t,X):
    dX=np.sin(t)*t
    return dX
def somefuncode(X,t):
    dX=np.sin(t)*t
    return dX
Xinitial=2.0
Tstart=0
Tend=200.0
SS=0.001
T=np.arange(Tstart,Tend,SS)
print("\nStarting RKCK naive solver...\n")
Xsol1=solver_rkck_naive(Xinitial,[Tstart,Tend],SS,somefun)
#plt.plot(T,Xsol1,'b',label='Naive')
#Xcomp=solver_rk4(Xinitial,[Tstart,Tend],SS,somefun)
#plt.plot(T,Xcomp,'g',label='RK4')
print("\nStarting RKCK adaptive solver...\n")

Xsol2=solver_rkck_adaptive(Xinitial,[Tstart,Tend],SS,somefun)
plt.plot(Xsol2[0],Xsol2[1],'ko--',label='Adaptive RK4')

s=time.clock()
DX=odeint(somefuncode,Xinitial,T)
print("TIME TAKEN FOR ODEINT = ",time.clock()-s)

plt.plot(T,DX,'r',label='ODEINT')
#plt.plot(T,np.exp(T)*Xinitial,'k',label='Soln')

plt.legend()
plt.show()
