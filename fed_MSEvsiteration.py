import time

start = time.time()

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
#N=int(input("enter the number of unknowns(N):  "))
N=200


# start = 0
# end = 20
# step = 0.2
# SNR = []

# value = start
# while value <= end:
#     SNR.append(value)
#     value += step

SNR=[5]
colours=['b','g','r','c','m']

#num_agents=list(range(1, 11, 1))
num_agents=[1,2,4,10,100]
count=0

for snrs in SNR:
    MSE_final=[]

    for agents in num_agents:

        #rho=float(input("enter the mask value(rho):  "))
        #reg=float(input("enter the regularization strength(reg):  "))
        rho=1
        reg=1

        A=[]
        gamma_w=[]
        M=[]
        y=[]

        #mean and precision of prior 
        m=np.zeros((N,1))
        gamma_x=1

        x_02= np.random.normal(0,1,size=(N,1))
        #mask=np.random.binomial(1.0, rho, (N,1))
        x = x_02


        #Denoisinh

        x_hat_1 = [np.random.normal(0.0, 1.0, (N,1))]
        alpha_1 = []
        eta_1 = []
        gamma_2=[]
        r_2=[]

        #LMMSE 
        x_hat_2 = []
        alpha_2 = []
        eta_2 = []
        gamma_1 = []
        r_1 = []

        #for momentum - regularization
        old_r_2=[]
        old_r_1=[]
        old_gamma_2=[]
        old_gamma_1=[]

        #n= int(input("enter the number of agents : "))
        n=agents
        print(n)
        for i in range (n):
            
            #size_M= int(input("enter the size of A_%d : "%i))
            size_M=int((N/2)//n)
            #snr=int(input("enter the SNR_%d : "%i))
            #SNR.append(snr)
            M.append(size_M)
            A.append(np.random.normal(0,1,size=(M[i], N)))
            noise_var=(np.mean((np.matmul(A[i],x))**2)*10**(-snrs/10))
            gamma_w.append(noise_var**-2)
            y.append(np.matmul(A[i],x) + np.random.normal(0.0, noise_var, (M[i],1)))

            #denoising
            gamma_2.append(1)
            r_2.append(np.random.normal(0.0, 1.0, (N,1)))
            
            #LMMSE
            x_hat_2.append(np.random.normal(0.0, 1.0, (N,1)))
            alpha_2.append(1)
            eta_2.append(1)
            gamma_1.append(1)
            r_1.append(np.random.normal(0.0, 1.0,(N,1)))

            old_r_2.append(np.zeros((N,1)))
            old_r_1.append(np.zeros((N,1)))
            old_gamma_2.append(1)
            old_gamma_1.append(1)


        r_new=[]

        gamma_temp=[]
        r_temp=[]
        gamma_x=1
        gamma_post=[]
        MSE_final=[]
        for i in range (20):

            #Denoising Updation 

            temp_g=0
            temp_r=0
            for k in range(n):
                temp_g=temp_g+gamma_1[k]
                temp_r=temp_r+r_1[k]*gamma_1[k]
            gamma_temp.append((temp_g))
            r_temp.append((gamma_temp[i]**-1)*(temp_r))

            gamma_post.append((gamma_x+gamma_temp[i]))
            x_hat_1.append(rho*(gamma_post[i]**-1)*(m*gamma_x + r_temp[i]*gamma_temp[i]))
            alpha_1.append(rho*(gamma_post[i]**-1)*gamma_temp[i])
            eta_1.append(gamma_temp[i]/alpha_1[i])

            MSE_f=0
            #LMMSE updation
            for j in range (n):

                gamma_2[j]= (eta_1[i] - gamma_1[j]) 
                r_2[j]= ((eta_1[i]*x_hat_1[i+1] - gamma_1[j]*r_1[j])/gamma_2[j])

                r_2[j]= ((1-reg)*old_r_2[j] + reg*r_2[j])
                old_r_2[j]=r_2[j]

                gamma_2[j]= ((1-reg)*old_gamma_2[j] + reg*gamma_2[j])
                old_gamma_2[j]=gamma_2[j]
                x_hat_2[j]= (np.matmul((np.linalg.inv(gamma_w[j]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N))),(gamma_w[j]*np.matmul(np.transpose(A[j]),y[j]) + gamma_2[j]*r_2[j])))
                alpha_2[j]= gamma_2[j]*np.trace(np.linalg.inv(gamma_w[j]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N)))/N
                eta_2[j]= gamma_2[j]/alpha_2[j]


                gamma_1[j]= eta_2[j] - gamma_2[j]
                r_1[j]= (eta_2[j]*x_hat_2[j] - gamma_2[j]*r_2[j])/gamma_1[j]
                r_1[j]= ((1-reg)*old_r_1[j] + reg*r_1[j])
                old_r_1[j]=r_1[j]
                gamma_1[j]= ((1-reg)*old_gamma_1[j] + reg*gamma_1[j])
                old_gamma_1[j]=gamma_1[j]        
            
                MSE_f =MSE_f+(np.mean((np.matmul(A[j],x_02) - np.matmul(A[j],x_hat_2[j]))**2)/np.mean(np.matmul(A[j],x_02)**2))
            MSE=MSE_f/n
            print("mse ", MSE)

            #import pdb; pdb.set_trace()
        
            MSE_final.append(MSE)

        x_03=np.linspace(1,20,20)
        #print(x_03)
        #import pdb; pdb.set_trace()
        end = time.time()
        print(end-start)

        plt.title('MSE vs Num of iterations')
        plt.semilogy(x_03,MSE_final)
        plt.xlabel('number of iterations')
        plt.ylabel('MSE')

count=count+1
plt.legend(['agents_1','agents_2','agents_4','agents_10','agents_100'])
plt.show()
    
