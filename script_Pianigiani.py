# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:08:25 2024

@author: Lucas Pianigiani

"""

import process_data
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import funciones_fft
from scipy.io import wavfile


filename1 = 'respirador.wav'          # nombre de archivo
fs_1, data1 = wavfile.read(filename1)   # frecuencia de muestreo y datos de la señal

# Definición de parámetro temporales
ts = 1 / fs_1                      # tiempo de muestreo
N_1 = len(data1)                   # número de muestras en el archivo de audio
t_1 = np.linspace(0, N_1 * ts, N_1)   # vector de tiempo
senial = data1[:, 1]            # se extrae un canal de la pista de audio (si el audio es estereo)
senial = senial * 3.3 / (2 ** 16 - 1) # se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)


filename2 = 'beep.wav'
fs2, data2 = wavfile.read(filename2)   # frecuencia de muestreo y datos de la señal
# Definición de parámetro temporales
ts_2 = 1 / fs2                     # tiempo de muestreo
N_2 = len(data2)                   # número de muestras en el archivo de audio
t_2 = np.linspace(0, N_2 * ts_2, N_2)   # vector de tiempo
senial2 = data2[:, 1]            # se extrae un canal de la pista de audio (si el audio es estereo)
senial2 = senial2 * 3.3 / (2 ** 16 - 1) # se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)


fig, axes = plt.subplots(2,2 , figsize=(20, 20))
fig.subplots_adjust(hspace=0.25)



axes[0][0].plot(t_1,data1,color ='blue')
axes[0][1].plot(t_2,data2,color = 'red')

axes[0][0].set_title('Data_1')
axes[0][0].grid()
axes[0][0].set_xlabel('Tiempo [s]', fontsize=10)
axes[0][0].set_ylabel('Amplitud[mV]', fontsize=10)

axes[0][1].set_title('Data_2')
axes[0][1].grid()
axes[0][1].set_xlabel('Tiempo [s]', fontsize=10)
axes[0][1].set_ylabel('Amplitud[mV]', fontsize=10)


#%%Analizar y graficar los espectros en frecuencia de ambas señales.
# (Nota: todos los pitidos de la alarma del respirador tienen las mismas componentes frecuenciales).

 
frec_1,espectro_1 = funciones_fft.fft_mag(senial,fs_1) # Calculo el espectro de frecuencia del respirador
frec_2,espectro_2 = funciones_fft.fft_mag(senial2,fs2) # Calculo el espectro de frecuencia de la alarma


axes[1][0].plot(frec_1,espectro_1,color ='blue')
axes[1][1].plot(frec_2,espectro_2,color = 'red')


axes[1][0].set_title('Espectro Data_1')
axes[1][0].grid()
axes[1][0].set_xlabel('Frecuencias [Hz]', fontsize=10)
axes[1][0].set_ylabel('Amplitud', fontsize=10)
axes[1][0].set_ylabel('Amplitud', fontsize=10)
axes[1][0].set_xlim([0,1400])

axes[1][1].set_title(' Espectro Data_2')
axes[1][1].grid()
axes[1][1].set_xlabel('Frecuencias [Hz]', fontsize=10)
axes[1][1].set_ylabel('Amplitud', fontsize=10)
axes[1][1].set_ylabel('Amplitud', fontsize=10)
axes[1][1].set_xlim([3200,4000])
plt.show()


#%% 2-  Determinar la función de transferencia H(s) de un filtro pasa bajos que permita recuperar la frecuencia fundamental y
   # los dos armónicos siguientes del sonido de alarma del respirador. La misma debe presentar una atenuación menor a 1dB en 
   # la banda de paso y debe asegurar una atenuación de al menos 40dB para la frecuencia fundamental del sonido del monitor multiparamétrico. 
   # Graficar la  magnitud de la respuesta en frecuencia de la función de transferencia propuesta.


#Habiendo graficado los espectros de frecuencias de las dos señales obtengo los siguientes parámetros a tener en cuenta para el disño del filtro 
# - La frecuencia fundamental y los dos primeros armónicos se ubican en  400 a 1325 Hz del respirador
# - La frecuencia fundamental de la alarma es de 3800 hz

# Ahora calculo el orden necesario del filtro para cumplir con la atenuación de 40db en la frecuencia funtamental de la alarma 
# Haciendo las pruebas para el orden del filtro, obtengo que un filtro de orden 4 tipo Chebyshev es de -45 db
# con una ateniación maxima de 0.5 db en la banda de paso por lo que se considera aceptable.

# ACLARACIÓN: En el software los valores de resistencias y capacitores son los ideales y en las simulaciones se reemplaaron por los reales mas cercanos.

#%%  Los ejerciios 3 y 4 se resuelven con los scripts provistos por la catedra "Diseño_filtros_analógicos" y "Implementacion_filtros_analógicos"

#Se adjuntan los filtros realizados en LTSpice
 
#%% 5-Proponer una frecuencia de muestreo que resulte apropiada para la aplicación (distinta a los 44,1kHz de los audios de prueba).
 # Analice qué tan adecuado es el filtro analógico propuesto en el punto 2 como anti alias (suponer CAD de 16bits y Vref de 3.3V).


# Calulo la atenuación requerida a en la frecuencia fundamental de la alarma.

delta_voltaje = 3.3 / ((2 **16) - 1)
fm_nueva = 3000  # Se propone una nueva frecuencia de muestreo de 3kHz

fm_s2 = int(fm_nueva/2)


inicio = np.where(frec_2 >= fm_s2)[0][0]
final = int(N_2/2)

amplitud_max = max(espectro_2[inicio: final])
atenuacion = 20 * np.log10 (amplitud_max / delta_voltaje)

posicion_relativa = np.argmax(espectro_2[inicio: final])
posicion_real = (posicion_relativa * fs2)/ N_2 + fm_s2

print("atenuacion requerida en la senial  es de " +  str(atenuacion) + "[db] a una frecuencia de " + str (posicion_real))

#En base a la atenuación requerida, se puede concluir que el filtro requerido NO será  suficiente para utilizar como antialias.
# Dado que se requiere una atenuación de -56 [db] a 3800Hz y el filtro obtenido tiene una atenuación de 45db


#%% 6- Diseñe, implemente y aplique un filtro IIR pasa alto que permita recuperar la señal del respirador a partir de la componente fundamental.
#   (No remuestrar)

# Para el diseño del filtro considero una fc = 400hz y una frecuencia de banda de rechazo de 100hz con una atenuacion de 30db


# Cargo el filtro 
filtro_iir = np.load('Filtro_pasa_alto_butter_fc400.npz', allow_pickle=True)



# Se extraen los coeficientes de numerador y denominador
Num_iir, Den_iir = filtro_iir['ba'] 

# Se aplica el filtrado iir
senial_iir = signal.lfilter(Num_iir, Den_iir, senial)


# Se calculan y grafican sus espectros (normalizados)
f1_iir, senial_iir_fft_mod = funciones_fft.fft_mag(senial_iir, fs_1)

fig, axes = plt.subplots(2,2, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5)

axes[0][0].plot(t_1,senial,color ='blue')
axes[0][1].plot(frec_1,espectro_1,color = 'red')

axes[1][0].plot(t_1,senial_iir,color ='blue')
axes[1][1].plot(f1_iir,senial_iir_fft_mod,color = 'red')


axes[0][0].set_title('Data_1')
axes[0][0].grid()
axes[0][0].set_xlabel('Tiempo [s]', fontsize=10)
axes[0][0].set_ylabel('Amplitud[mV]', fontsize=10)

axes[0][1].set_title('Espectro Data_1')
axes[0][1].grid()
axes[0][1].set_xlabel('Tiempo [s]', fontsize=10)
axes[0][1].set_ylabel('Amplitud[mV]', fontsize=10)
axes[0][1].set_xlim([0,1500])

axes[1][0].set_title(' Data_1 Filtrada')
axes[1][0].grid()
axes[1][0].set_xlabel('Frecuencias [Hz]', fontsize=10)
axes[1][0].set_ylabel('Amplitud', fontsize=10)
axes[1][0].set_ylabel('Amplitud', fontsize=10)


axes[1][1].set_title(' Espectro Data_1 Filtrada')
axes[1][1].grid()
axes[1][1].set_xlabel('Frecuencias [Hz]', fontsize=10)
axes[1][1].set_ylabel('Amplitud', fontsize=10)
axes[1][1].set_ylabel('Amplitud', fontsize=10)
axes[1][1].set_xlim([0,1500])
plt.show()