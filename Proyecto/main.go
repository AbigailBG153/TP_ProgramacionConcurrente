package main

import (
	"SVM-TP/MSV"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/schollz/progressbar/v3"
)

// Estructura para representar el modelo SVM
type SVM struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Lambda       float64
}

// Función para entrenar el SVM usando descenso de gradiente
func (svm *SVM) Train(X [][]float64, y []float64, epochs int) {
	nSamples, nFeatures := len(X), len(X[0])

	// Inicializar pesos y sesgo
	svm.Weights = make([]float64, nFeatures)
	svm.Bias = 0.0

	// Crear la barra de progreso para las épocas
	bar := progressbar.Default(int64(epochs), "Entrenando SVM")

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < nSamples; i++ {
			// Calcular la condición para actualizar (y * (X · W + b) >= 1)
			if y[i]*dotProduct(X[i], svm.Weights)+svm.Bias >= 1 {
				// No penalizar
				for j := 0; j < nFeatures; j++ {
					svm.Weights[j] -= svm.LearningRate * (2 * svm.Lambda * svm.Weights[j])
				}
			} else {
				// Penalización
				for j := 0; j < nFeatures; j++ {
					svm.Weights[j] -= svm.LearningRate * (2*svm.Lambda*svm.Weights[j] - X[i][j]*y[i])
				}
				svm.Bias -= svm.LearningRate * y[i]
			}
		}
		bar.Add(1) // Actualizar la barra de progreso después de cada época
	}
}

// Función para predecir una muestra con el SVM entrenado
func (svm *SVM) Predict(X [][]float64) []float64 {
	predictions := make([]float64, len(X))
	for i, sample := range X {
		if dotProduct(sample, svm.Weights)+svm.Bias >= 0 {
			predictions[i] = 1
		} else {
			predictions[i] = -1
		}
	}
	return predictions
}

// Producto punto entre dos vectores
func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Función para calcular la precisión
func AccuracyMetric(actual []float64, predicted []float64) float64 {
	correct := 0
	for i := range actual {
		if actual[i] == predicted[i] {
			correct++
		}
	}
	return (float64(correct) / float64(len(actual))) * 100.0
}

func main() {
	rand.Seed(time.Now().UnixNano())

	filepath := "Data/dataset.csv"
	blockSize := 1000 // Ajusta este valor según tus necesidades

	fmt.Println("Iniciando carga de datos...")
	startTime := time.Now()
	data, labels, err := MSV.ReadCSVConcurrently(filepath, blockSize)
	if err != nil {
		log.Fatalf("Error al leer el CSV: %v", err)
	}
	loadDataTime := time.Since(startTime)
	fmt.Printf("Carga de datos completada en: %s\n", loadDataTime)

	// Convertir etiquetas para SVM (se espera -1 o 1)
	for i := 0; i < len(labels); i++ {
		if labels[i] == 0 {
			labels[i] = -1
		} else {
			labels[i] = 1
		}
	}

	fmt.Println("Iniciando separación de datos...")
	startTime = time.Now()
	X_train, X_test, y_train, y_test := MSV.TrainTestSplit(data, labels, 0.2)
	separationTime := time.Since(startTime)
	fmt.Printf("Separación de datos completada en: %s\n", separationTime)

	// Definir hiperparámetros del SVM
	svm := SVM{
		LearningRate: 0.001,
		Lambda:       0.01,
	}

	fmt.Println("Iniciando Entrenamiento...")
	startTrain := time.Now()
	svm.Train(X_train, y_train, 1000) // Número de épocas ajustado
	trainDuration := time.Since(startTrain)
	fmt.Printf("Tiempo de entrenamiento: %v\n", trainDuration)

	fmt.Println("Iniciando Predicción...")
	startPredict := time.Now()
	predictions := svm.Predict(X_test)
	predictDuration := time.Since(startPredict)
	fmt.Printf("Tiempo de predicción: %v\n", predictDuration)

	accuracy := AccuracyMetric(y_test, predictions)
	fmt.Printf("Precisión: %.2f%%\n", accuracy)
}
