package modelo

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// Parámetros del modelo SVM
// Parámetros del modelo SVM
type SVM struct {
	Weights []float64 // Exported by capitalizing
	Bias    float64   // Exported by capitalizing
	Alpha   float64   // Tasa de aprendizaje, exported by capitalizing
	Epochs  int       // Número de iteraciones, exported by capitalizing
}

// Función para dividir los datos en conjunto de entrenamiento y prueba
func SplitData(features [][]float64, labels []float64, ratio float64) ([]DataPoint, []DataPoint) {
	start := time.Now() // Medir tiempo de inicio
	var trainSet []DataPoint
	var testSet []DataPoint

	for i := range features {
		dataPoint := DataPoint{
			features: features[i],
			label:    labels[i],
		}
		if rand.Float64() < ratio {
			trainSet = append(trainSet, dataPoint)
		} else {
			testSet = append(testSet, dataPoint)
		}
		// Mostrar barra de progreso
		PrintProgressBar(i+1, len(features), "Dividiendo los datos")
	}
	fmt.Printf("\nTiempo de división de los datos: %v\n", time.Since(start))
	return trainSet, testSet
}

// Función para entrenar el SVM con concurrencia
func (svm *SVM) Train(data []DataPoint) {
	start := time.Now() // Medir tiempo de inicio
	var wg sync.WaitGroup
	n := len(data)
	featureLen := len(data[0].features)
	svm.Weights = make([]float64, featureLen)

	for epoch := 0; epoch < svm.Epochs; epoch++ {
		wg.Add(n)

		// Goroutine para cada punto de datos
		for i := 0; i < n; i++ {
			go func(i int) {
				defer wg.Done()
				svm.UpdateWeights(data[i])
			}(i)
		}

		// Sincronización: esperar a que todas las goroutines terminen
		wg.Wait()

		// Mostrar barra de progreso del entrenamiento
		PrintProgressBar(epoch+1, svm.Epochs, "Entrenando el modelo")
	}
	fmt.Printf("\nTiempo de entrenamiento: %v\n", time.Since(start))
	fmt.Println("Entrenamiento finalizado. Pesos:", svm.Weights, "Bias:", svm.Bias)
}

// Función que actualiza los pesos para cada punto de datos
func (svm *SVM) UpdateWeights(point DataPoint) {
	prediction := svm.Predict(point.features)
	// Verificación de la condición de margen
	if point.label*prediction < 1 {
		// Actualizar los pesos
		for j := 0; j < len(svm.Weights); j++ {
			svm.Weights[j] += svm.Alpha * (point.label*point.features[j] - 2*1.0/float64(len(point.features))*svm.Weights[j])
		}
		// Actualizar el sesgo (Bias)
		svm.Bias += svm.Alpha * point.label
	}
}

// Función para hacer predicciones con el SVM
func (svm *SVM) Predict(features []float64) float64 {
	dotProduct := 0.0
	for i := 0; i < len(svm.Weights); i++ {
		dotProduct += svm.Weights[i] * features[i]
	}
	return dotProduct + svm.Bias
}

// Función para evaluar el modelo SVM en el conjunto de prueba
// Función para evaluar el modelo SVM en el conjunto de prueba
func (svm *SVM) Evaluate(testSet []DataPoint) float64 {
	start := time.Now() // Medir tiempo de inicio
	var correct int
	for i, point := range testSet {
		prediction := svm.Predict(point.features)
		if (prediction >= 0 && point.label == 1) || (prediction < 0 && point.label == -1) {
			correct++
		}
		// Mostrar barra de progreso durante la evaluación
		PrintProgressBar(i+1, len(testSet), "Evaluando el modelo")
	}
	accuracy := float64(correct) / float64(len(testSet))
	fmt.Printf("\nTiempo de evaluación: %v\n", time.Since(start))
	fmt.Printf("Precisión del modelo en el conjunto de prueba: %.2f%%\n", accuracy*100)
	return accuracy // Ahora devuelve la precisión
}

// Función para imprimir la barra de progreso
func PrintProgressBar(iteration, total int, task string) {
	percent := float64(iteration) / float64(total) * 100
	barLength := 40 // Longitud de la barra de progreso
	filledLength := int(barLength * iteration / total)
	bar := ""
	for i := 0; i < filledLength; i++ {
		bar += "="
	}
	for i := filledLength; i < barLength; i++ {
		bar += " "
	}
	fmt.Printf("\r%s [%s] %.2f%%", task, bar, percent)
	if iteration == total {
		fmt.Println()
	}
}
