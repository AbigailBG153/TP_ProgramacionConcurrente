package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

// Datos de entrenamiento: características y etiquetas
type DataPoint struct {
	features []float64
	label    float64
}

// Parámetros del modelo SVM
type SVM struct {
	weights []float64
	bias    float64
	alpha   float64 // Tasa de aprendizaje
	epochs  int     // Número de iteraciones
}

// Función para cargar el archivo X.csv (ignorando la primera fila de cabeceras)
func loadFeatures(filename string) ([][]float64, error) {
	start := time.Now() // Medir tiempo de inicio
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error abriendo el archivo: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error leyendo el archivo CSV: %v", err)
	}

	// Ignorar la primera fila (cabeceras)
	records = records[1:]

	features := make([][]float64, len(records))

	for i, record := range records {
		features[i] = make([]float64, len(record))
		for j, value := range record {
			floatVal, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, fmt.Errorf("error convirtiendo a float64: %v", err)
			}
			features[i][j] = floatVal
		}
		// Mostrar barra de progreso
		printProgressBar(i+1, len(records), "Cargando características")
	}
	fmt.Printf("\nTiempo de carga de características: %v\n", time.Since(start))
	return features, nil
}

// Función para cargar el archivo y.csv (ignorando la primera fila de cabeceras)
func loadLabels(filename string) ([]float64, error) {
	start := time.Now() // Medir tiempo de inicio
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error abriendo el archivo: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error leyendo el archivo CSV: %v", err)
	}

	// Ignorar la primera fila (cabeceras)
	records = records[1:]

	labels := make([]float64, len(records))
	for i, record := range records {
		if len(record) != 1 {
			return nil, fmt.Errorf("esperaba una única columna en el archivo de etiquetas en la fila %d", i+1)
		}
		label, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, fmt.Errorf("error convirtiendo la etiqueta a float64: %v", err)
		}
		labels[i] = label
		// Mostrar barra de progreso
		printProgressBar(i+1, len(records), "Cargando etiquetas")
	}
	fmt.Printf("\nTiempo de carga de etiquetas: %v\n", time.Since(start))
	return labels, nil
}

// Función para dividir los datos en conjunto de entrenamiento y prueba
func splitData(features [][]float64, labels []float64, ratio float64) ([]DataPoint, []DataPoint) {
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
		printProgressBar(i+1, len(features), "Dividiendo los datos")
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
	svm.weights = make([]float64, featureLen)

	for epoch := 0; epoch < svm.epochs; epoch++ {
		wg.Add(n)

		// Goroutine para cada punto de datos
		for i := 0; i < n; i++ {
			go func(i int) {
				defer wg.Done()
				svm.updateWeights(data[i])
			}(i)
		}

		// Sincronización: esperar a que todas las goroutines terminen
		wg.Wait()

		// Mostrar barra de progreso del entrenamiento
		printProgressBar(epoch+1, svm.epochs, "Entrenando el modelo")
	}
	fmt.Printf("\nTiempo de entrenamiento: %v\n", time.Since(start))
	fmt.Println("Entrenamiento finalizado. Pesos:", svm.weights, "Bias:", svm.bias)
}

// Función que actualiza los pesos para cada punto de datos
func (svm *SVM) updateWeights(point DataPoint) {
	prediction := svm.predict(point.features)
	// Verificación de la condición de margen
	if point.label*prediction < 1 {
		// Actualizar los pesos
		for j := 0; j < len(svm.weights); j++ {
			svm.weights[j] += svm.alpha * (point.label*point.features[j] - 2*1.0/float64(len(point.features))*svm.weights[j])
		}
		// Actualizar el sesgo (bias)
		svm.bias += svm.alpha * point.label
	}
}

// Función para hacer predicciones con el SVM
func (svm *SVM) predict(features []float64) float64 {
	dotProduct := 0.0
	for i := 0; i < len(svm.weights); i++ {
		dotProduct += svm.weights[i] * features[i]
	}
	return dotProduct + svm.bias
}

// Función para evaluar el modelo SVM en el conjunto de prueba
func (svm *SVM) Evaluate(testSet []DataPoint) {
	start := time.Now() // Medir tiempo de inicio
	var correct int
	for i, point := range testSet {
		prediction := svm.predict(point.features)
		if (prediction >= 0 && point.label == 1) || (prediction < 0 && point.label == -1) {
			correct++
		}
		// Mostrar barra de progreso durante la evaluación
		printProgressBar(i+1, len(testSet), "Evaluando el modelo")
	}
	accuracy := float64(correct) / float64(len(testSet))
	fmt.Printf("\nTiempo de evaluación: %v\n", time.Since(start))
	fmt.Printf("Precisión del modelo en el conjunto de prueba: %.2f%%\n", accuracy*100)
}

// Función para imprimir la barra de progreso
func printProgressBar(iteration, total int, task string) {
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

func main() {
	rand.Seed(time.Now().UnixNano())

	// Cargar las características desde X.csv
	X, err := loadFeatures("./Data/X.csv")
	if err != nil {
		log.Fatalf("Error al cargar características: %v", err)
	}

	// Cargar las etiquetas desde y.csv
	y, err := loadLabels("./Data/y.csv")
	if err != nil {
		log.Fatalf("Error al cargar etiquetas: %v", err)
	}

	// Dividir los datos en conjunto de entrenamiento y prueba
	trainSet, testSet := splitData(X, y, 0.7) // Usar 70% para entrenamiento y 30% para prueba

	// Crear el modelo SVM
	svm := SVM{
		alpha:  0.01, // Tasa de aprendizaje
		epochs: 1000, // Número de iteraciones
	}

	// Entrenar el modelo
	svm.Train(trainSet)

	// Evaluar el modelo en el conjunto de prueba
	svm.Evaluate(testSet)
}
