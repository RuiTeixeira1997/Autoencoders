# -*- coding: utf-8 -*-
"""

@author: Rui Teixeira

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import decomposition
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import abc
import six
import math
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.optimizers import Adam
from numpy import linalg as LA
import pickle
from numpy.linalg import matrix_rank

K=np.array([[1,2,1,0.5,2.5],[1.5,0.5,1,0,2],[1,0.5,2,2,5]])

def AnalyticalPCA(y, dimension):
    pca = PCA(n_components=dimension)
    pca.fit(y)
    loadings = pca.components_

    return loadings

def generate_sequence(length, n_features):
    '''
    Parameters
    ----------
    length : Tamanho da atributos, corresponde ao i
    n_features : Número máximo e mínimo dos valores de base de dados, por exemplo
    se n_features = 5 vai gerar valores entre [-5,5]
    Returns
    -------
    list : Vetor de dimensão lenght com valores entre [-n_features,n_features]
    '''
    return [random.uniform(-n_features, n_features) for _ in range(length)]

def criar_DB(Tamanho_da_DB,Tamanho_das_amostras,Intervalo,ficheiro):
    '''
    Gerar base de dados para teste nos autoenconders e guarda-la num ficheiro.
    
    Parameters
    ----------
    Tamanho_da_DB : Corresponde ao n
    Tamanho_das_amostras : Corresponde ao I
    Intervalo : Intervalo para gerar os valores dos vetores 
    ficheiro : Nome do ficheiro

    '''
    X=[]
    for i in range(Tamanho_da_DB):
        a=generate_sequence(Tamanho_das_amostras,Intervalo)
        X.append(a)
    pickle.dump( np.asarray(X), open( ficheiro, "wb" ) )
    return np.array(X, dtype='f')

def criar_DB1(Tamanho_da_DB,Tamanho_das_amostras,Intervalo,ficheiro):
    '''
    Caso especial para os valores proprios serem muito distintos em tamanho de grandeza

    '''
    X=[]
    for i in range(Tamanho_da_DB):
        a=generate_sequence(Tamanho_das_amostras,Intervalo)
        a[0]=a[0]*100
        a[1]=a[1]*20
        a[2]=a[2]*10
        a[3]=a[3]
        a[4]=a[4]*(1/5)
        X.append(a)
    pickle.dump( np.asarray(X), open( ficheiro, "wb" ) )
    return np.array(X, dtype='f')

def load_DB(ficheiro):
    '''
    Carregar base de dados atrvés de um ficheiro

    '''
    return pickle.load( open(ficheiro, "rb" ) )

def centrar_DB(X,A='F'):
    '''
    Função que recebe uma base de dados e vai centrar a mesma.
    Caso A=='T' devolve o vetor da média.
    '''
    P=np.array(X, dtype='f')
    media=[]
    for j in range(len(P[0])):
        media.append(np.mean(P[:,j]))
    for i in range(X.shape[0]):
        P[i]=P[i] - media
    if A=='T':
        print('vetor_media= ', media)
    
    return P


def loss_f(U,V,a,b,x):
    '''
    Função de loss contruída manualmente
    '''
    k=x
    z=np.matmul(U,x)+a
    xn=np.matmul(V,z)+b
    for i in range(len(k)):
        k[i]=k[i]*k[i]
        xn[i]=xn[i]*xn[i]
    erro1=k-xn
    return np.sum(erro1)/len(k)

def loss_TF(U,V,a,b,x):
    '''
    Função de loss baseada no tensorflow
    '''
    z=np.matmul(U,x)+a
    xx=np.matmul(V,z)+b
    y_pred = ops.convert_to_tensor(xx)
    y_true = math_ops.cast(x, y_pred.dtype)
    return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)

def loss_database(U,V,a,b,database):
    '''
    Dados os pesos do autoencoder, calcula o valor de loss para todos os elementos da base
    de dados fazendo uso da função de loss construída manualmente
    '''
    
    err2=0
    for x in database:
        r=loss_f(U,V,a,b,x)
        err2=err2+r
        
    j=err2/len(database)
    return j

def loss_databaseTF(U,V,a,b,database):
    '''
    Dados os pesos do autoencoder, calcula o valor de loss para todos os elementos da base
    de dados fazendo uso da função de loss baseada no tensorflow
    '''
    
    err3=0
    for x in database:
        r=loss_TF(U,V,a,b,x)
        err3=err3+r
    return err3/len(database)

def Teste(V):
    '''
    Função que vai testar se (V^TV)^-1V^T=U
    '''
    z=np.matmul(np.transpose(V),V)
    k=LA.inv(z)
    j=np.matmul(k,np.transpose(V))
    return j

def Teste1(V,a,b):
    '''
    Verificar se -b -Va dá zero
    
    Returns
    -------
    z : O valor de Va
    b : O valor de b
    -b -z : O valor de -b -Va que queremos verificar se está perto de zero
    k : o valor máximo em modulo de -b-Va
    '''
    z=np.matmul(V,a)
    k=-b-z
    return z , b , k, np.max(np.abs(k))


def XTX(x):
    '''
    Recebendo uma matriz A,retorna o valor de A^TA.
    '''
    return np.matmul(np.transpose(x),x)

def Teste2(X,U):
    '''
    Verificar se V=X^{T}XU^{T}(UX^{T}XU^{T})^{-1} 

    '''
    O=XTX(X)
    O1=np.matmul(O,np.transpose(U))
    O2=np.matmul(U,O)
    O3=np.matmul(O2,np.transpose(U))
    k=LA.inv(O3)
    FINAL=np.matmul(O1,k)
    return FINAL

def produtoInterno(x,y,axes):
    '''
    Calcula o produto interno entre dois vetores
    '''
    return tf.tensordot(x,y,axes)

def Operations(x):
    '''
    Dado como input uma matriz A, devolve os valores e vetores próprios da matriz A^TA.
    '''
    xTx=np.matmul(np.transpose(x),x)
    w, v = LA.eig(xTx)
    return w,v


def OrdenaVal(C):
    '''
    Função que recebendo uma matriz A, devolve os valores e vetores próprios ordenados de A^TA
    '''
    val,vec=Operations(C)
    idx = val.argsort()[::-1]
    val = val[idx]
    vec = vec[:,idx]
    return val,vec

def VerifyOrt(U,C,mostrar='F'):
    '''
    Função que vai verificar a ortogonalidade entre os vetores de U e os últimos
    vetores próprios da matriz C^TC
    
    Retorna o desvio à ortogonalidade
    '''
    C=np.array(C, dtype='f')
    val,vec=OrdenaVal(C)
    if mostrar=='T':
        print('Valores próprios:')
        print(val)
        print('Vetores próprios:')
        print(vec)
    J=len(C[0])-len(U)
    A=[]
    for i in U:
        B=[]
        for p in range(J):
            q=produtoInterno(i,np.transpose(vec[:,J+p+1]),1)
            B.append(q/(np.linalg.norm(i,2)*np.linalg.norm(vec[:,J+p+1],2)))
        A.append(B)
    return A , np.max(np.abs(A))

def Relu(vec):
    '''
    recebe um vetor como input e deolve o vetor correspondente ao calculo da ReLU
    para esse mesmo vetor
    '''
    return tf.nn.relu(vec).numpy()

def Find_pattern_Enc(X,U,a):
    '''
    Função que procura padrões ao fazer encode dos dados
    '''
    contador=np.zeros(len(U))
    for i in range(len(X)):
        k=Relu(np.matmul(U,X[i])+a)
        for j in range(len(U)):
            if k[j]==0:
                contador[j]+=1
    perc=np.zeros(len(U))
    for i in range(len(contador)):
        perc[i]=(contador[i]/len(X))*100
    return contador,perc


def Find_pattern_Dec(X,U,V,a,b):
    '''
    Função que procura padrões ao fazer decode dos dados
    '''
    contador=np.zeros(len(V))
    for i in range(len(X)):
        k=Relu(np.matmul(U,X[i])+a)
        z=Relu(np.matmul(V,k)+b)
        for j in range(len(V)):
            if z[j]==0:
                contador[j]+=1
    perc=np.zeros(len(V))
    for i in range(len(contador)):
        perc[i]=(contador[i]/len(X))*100
    return contador,perc
        
    
def get_par(X,I,J):
    '''
    Função que retorna parâmetros específios de U e V muito usados ao longo
    do documento
    '''
    val,vec=OrdenaVal(X)
    V=vec[:,:J]
    U=np.transpose(V)
    return U ,V

def DB_Generator(tipo,I,J):
    '''
    Parameters
    ----------
    tipo : É o tipo do gerador:
        tipo 1 = Base canónica
        tipo 2 = Base em que o vetor i tem componentes todas 1's excepto na posição i que é zero a multiplicar por
                 1*sqrt(I-1)
        tipo 3 = Gerar um valor aleatorio para o primeiro vetor e de seguidas os outros vão ser todos zeros excepto na
                 primeira posição que vai ser o elemento j do primeiro vetor e na posição i vamos ter - a primeiro
                 elemento do vetor aleatorio
    I : Número de atributos
    J : Número a reduzir a dimensão
    Returns
    -------
    A : Devolve o gerador

    '''
    if tipo==1:
        A=[]
        for i in range(J):
            p=np.zeros(I)
            p[i]=1
            A.append(p)
        return A
            
    if tipo==2:
        A=[]
        for i in range(J):
            p=np.ones(I)
            p[i]=0
            k=1/math.sqrt(I-1)
            A.append(k* p)
        return A
    if tipo==3:
        e=generate_sequence(I,10)
        A=[]
        A.append(np.array(e))
        for i in range(J-1):
            p=np.zeros(I)
            p[0]=e[i+1]
            p[i+1]=-e[0]
            A.append(p)
        return A

def DB_type(tipo,ficheiro,N,I,J):
    '''

    Parameters
    ----------
    tipo : Tipo de gerador, opções : 1 ,2 e 3
    N : Tamanho do Dataset
    I : Número de Atributos
    J : Valor a reduzir no autoenconder

    Returns
    -------
    
    Um dataset com N amostras.

    '''
    gerador= DB_Generator(tipo,I,J)
    L=[]
    for i in range(N):
        j=0
        Val=np.zeros(I)
        Lambdas=generate_sequence(J, 10)
        for ele in gerador:
            
            Val+= (ele*Lambdas[j])
            j=j+1
        L.append(Val)
    pickle.dump( np.array(L), open(ficheiro, "wb" ) )  
    return np.array(L)

def Base_Regular(N,I,J,ficheiro):
    '''
    Gerar uma base de dados regular
    
    '''
    X= DB_type(3,'Db_teste',10,I,J)
    vec,_=get_par(X,I,J)
    L=[]
    for i in range(N):
        j=0
        Val=np.zeros(I)
        Lambdas=generate_sequence(J, 10)
        print(Lambdas)
        for ele in vec:
            
            Val+= (ele*Lambdas[j])
            j=j+1
        L.append(Val)
    pickle.dump( np.array(L), open(ficheiro, "wb" ) )  
    return np.array(L)

def Base_Regular_positiva(N,I,J,ficheiro,ficheiro1):
    '''
    Gerar base de dados positiva (Todos os elementos são maiores que zero)
    e guardar num ficheiro a base de dados e os vetores que a geram
    '''
    vec=[[1,2,3,4,5],[2,3,4,5,1],[3,4,5,1,2]]
    vec=np.array(vec)
    vec.astype(float)
    L=[]
    for i in range(N):
        j=0
        Val=np.zeros(I)
        Lambdas=[random.uniform(0,50) for _ in range(J)]
        for ele in vec:
            
            Val+= (ele*Lambdas[j])
            j=j+1
        L.append(Val)
    pickle.dump( np.array(L), open(ficheiro, "wb" ) )  
    vec_norm=gs(vec)
    pickle.dump( vec_norm, open(ficheiro1, "wb" ) )  

    return vec_norm,np.array(L)

            
def D(U,Util):
    '''
    Métrica : Diferença máxima
    '''
    return np.mean(np.abs(U-Util))
        
def M(U,Util):
    '''
    Métrica : Média da diferença
    '''
    return np.max(np.abs(U-Util))

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = np.array2string(a, max_line_width=np.infty).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return rv

def sn(x,U,a):
    'Calculo de sn'
    s=np.zeros(len(U))
    k=Relu(np.matmul(U,x)+a)
    for i in range(len(s)):
        if k[i]>0:
            s[i]=1
        else:
            pass
    return s


def tn(x,U,V,a,b):
    'calculo de tn'
    t=np.zeros(len(V))
    z=Relu(np.matmul(U,x)+a)
    xx=Relu(np.matmul(V,z)+b)
    for i in range(len(t)):
        if xx[i]>0:
            t[i]=1
        else:
            pass
    return t

def Isn(x,U,a):
    'calculo da matriz Isn'
    s=sn(x,U,a)
    Is=np.zeros([len(U),len(U)])
    for i in range(len(U)):
        Is[i][i]=s[i]
    return Is

def Itn(x,U,V,a,b):
    'calculo da matriz Itn'
    t=tn(x,U,V,a,b)
    It=np.zeros([len(V),len(V)])
    for i in range(len(V)):
        It[i][i]=t[i]
    return It

def print_history_loss(history):
    '''
    Devolve o gráfico do valor de loss para o nº de épocas
    '''
    # print(history.history.keys())
    plt.plot(np.log10(history.history['loss']))
    plt.title('Loss do modelo')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def print_erro_loss(history,erro):
    '''
    Gráfico que compara os erros de L(X;U,V,a,b) e K(X;a,b)
    '''
    # print(history.history.keys())
    plt.plot(np.log10(history.history['loss']))
    plt.plot(np.log10(erro))
    plt.title('Loss do modelo')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['L(X;U,V,a,b)', 'K(X;a,b)'], loc='upper left')
    plt.show()
    
def create_Autoencoder(I,J,activ):
    '''
    Parameters
    ----------
    I : Valor de I
    J : Tamanho para o qual se vai reduzir os dados

    Returns
    -------
    autoencoder : Retorna o modelo do autoenconder
    '''
    input_i=Input(shape=(I,))
    encoded = Dense(units=J, activation=activ)(input_i)
    decoded = Dense(units=I, activation=activ)(encoded)
    encoder = Model(inputs=input_i, outputs=encoded) 
    autoencoder = Model(inputs=input_i, outputs=decoded)
    return autoencoder
    
def create_AutoencoderELU(I,J):
    '''
    Autoencoder com a função de ativação ELU
    '''
    input_i=Input(shape=(I,))
    encoded = Dense(units=J, activation=tf.nn.elu)(input_i)
    decoded = Dense(units=I, activation=tf.nn.elu)(encoded)
    encoder = Model(inputs=input_i, outputs=encoded) 
    autoencoder = Model(inputs=input_i, outputs=decoded)
    return autoencoder, encoder



def create_Autoencoder_regularizador(I,J,activ):
    '''
    Autoencoder que utiliza regularizadores
    '''
    input_i=Input(shape=(I,))
    encoded = Dense(units=J, activation=activ,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(input_i)
    decoded = Dense(units=I, activation=activ,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(encoded)
    encoder = Model(inputs=input_i, outputs=encoded) 
    autoencoder = Model(inputs=input_i, outputs=decoded)
    return autoencoder, encoder



def compiletrain(autoencoder, X_train,epocas):
    '''
    Função que realiza o treino do autoencoder
    '''
    checkpointer = ModelCheckpoint(filepath="best_weights_modelo.hdf5", monitor = 'loss', verbose=1, save_best_only=True)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = autoencoder.fit(X_train, X_train,epochs=epocas,shuffle=True,callbacks=[checkpointer])
    #history = autoencoder.fit(X_train, X_train,epochs=epocas,shuffle=True)
    return autoencoder,history



def compiletrain_bias_set(autoencoder, X_train,epocas):
    '''
    Função que permite inicializar o treino escolhe os parâmetros iniciais
    '''
    #checkpointer = ModelCheckpoint(filepath="best_weights_modelo.hdf5", monitor = 'loss', verbose=1, save_best_only=True)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    #history = autoencoder.fit(X_train, X_train,epochs=10000,shuffle=True,callbacks=[checkpointer])
    for i in range(epocas):
        history = autoencoder.fit(X_train, X_train,epochs=1,shuffle=True)
        V,a,U,b=autoencoder.get_weights()
        autoencoder.set_weights((V,[100,100,100],U,[100,100,100,100,100]))
    return autoencoder,history


def compiletrain_SET_ALL(autoencoder, X_train,U,V,a,b):
    '''
    Função que permite avaliar o erro do autoencoder com a função de loss
    do tensorflow com os parâmetros escolhidos pelo utlizador
    '''
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    autoencoder.set_weights((V,a,U,b))
    for i in range(2):
        history = autoencoder.fit(X_train, X_train,epochs=1,shuffle=True)
        autoencoder.set_weights((V,a,U,b))
    return autoencoder,history

def gradiente_a(x,U,V,a,b):
    
    '''
    Calculo do gradiente de a 
    '''
    
    Is=Isn(x,U,a)
    It=Itn(x,U,V,a,b)
    p=2*(np.matmul(np.transpose(x),np.transpose(U)))
    s=2*(np.matmul(np.transpose(x),It))
    ss=np.matmul(s,V)
    sss=np.matmul(ss,Is)
    t=2*(np.matmul(np.transpose(b),V))
    tt=np.matmul(t,Is)
    q=2*np.transpose(a)
    return a-sss+tt+q
    

def gradiente_b(x,U,V,a,b):
        
    '''
    Calculo do gradiente de b
    '''
    
    Is=Isn(x,U,a)
    It=Itn(x,U,V,a,b)
    p=2*(np.matmul(np.transpose(x),It))
    s=2*(np.matmul(np.transpose(x),np.transpose(U)))
    ss=np.matmul(s,np.transpose(Is))
    sss=np.matmul(ss,np.transpose(V))
    t=2*(np.matmul(np.transpose(a),np.transpose(Is)))
    tt=np.matmul(t,np.transpose(V))
    q=2*np.transpose(b)
    return -p +sss+tt+q



def loss_gradiente(X,a,b):
        
    '''
    Calculo do valor de loss da base de dados X com os parâmetros calculados
    a e b
    '''
    
    J=len(a)
    I=len(b)
    U,V=get_par(X,I,J)
    v_lossA=np.zeros(J)
    for x in X:
        v_lossA=v_lossA+gradiente_a(x,U,V,a,b)
    v_lossA=v_lossA*(1/len(X))
    v_lossB=np.zeros(I)
    for x in X:
        v_lossB=v_lossB+gradiente_b(x,U,V,a,b)
    v_lossB=v_lossB*(1/len(X))
    return v_lossA,v_lossB

        

def Metodo_gradiente(n_iteras,X,a,b,lr):
    '''
    Parameters
    ----------
    n_iteras : Nº de itreções do método do gradiente desenvolvido
    X : Base de dados
    a : valor de a inicial
    b : valor de b inicial
    lr : learning rate
    
    Returns
    -------
    a : o valor a depois de aplicado o método do gradiente
    b : o valor b depois de aplicado o método do gradiente
    erros : vetor dos erros

    '''
    U,V=get_par(X,5,3)
    erros=[]
    for i in range(n_iteras):
        v_lossA,v_lossB=loss_gradiente(X,a,b)
        a=a-lr*v_lossA
        b=b-lr*v_lossB
        erro=loss_databasenova(U,V,a,b,X)
        erros.append(erro)
        print(i,erro)
    return a,b,erros

def gs(X, row_vecs=True, norm = True):
    
    '''
    Função que calcula O Processo de ortogonalização de Gram-Schmidt
    '''
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
    
    
def overbar_a(X,U):
    
    '''
    Calculo de a_overbar
    '''
    
    a=np.zeros(len(U))
    val_max=-10000000000000000000
    for j in range(len(U)):
        val_max=-10000000000000000000
        for i in range(len(X)):
            z=np.matmul(U,X[i])
            if val_max < z[j]:
                a[j]=z[j]
                val_max=z[j]
    return a
        
def underbar_a(X,U):
    
    '''
    Calculo de a_underbar
    '''
    a=np.zeros(len(U))
    val_min=10000000000000
    for j in range(len(U)):
        val_min=10000000000000
        for i in range(len(X)):
            z=np.matmul(U,X[i])
            if val_min > z[j]:
                a[j]=z[j]
                val_min=z[j]
    return -a
        

def erro(X):
    '''
    Calculo do erro da secção 6.7
    '''
    val=0
    for i in range(15):
        for j in range(5):
            if X[i][j]<0:
                val=val+(X[i][j]*X[i][j])
    return val/(15*5)



