package Modelado

import (
	"sync"

	"github.com/gorilla/websocket"
)

// Estructura para representar el modelo SVM
type SVM struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Lambda       float64
	mutex        sync.Mutex // Mutex para evitar condiciones de carrera
}

// Función para entrenar el SVM usando descenso de gradiente de manera concurrente y enviar progreso por WebSocket
func (svm *SVM) Train(X [][]float64, y []float64, epochs int, conn *websocket.Conn) {
	nSamples, nFeatures := len(X), len(X[0])

	// Inicializar pesos y sesgo
	svm.Weights = make([]float64, nFeatures)
	svm.Bias = 0.0

	// Entrenar por cada época y enviar progreso
	for epoch := 0; epoch < epochs; epoch++ {
		var wg sync.WaitGroup

		// Crear las tareas para cada muestra de entrenamiento
		for i := 0; i < nSamples; i++ {
			wg.Add(1)

			// Ejecutar el entrenamiento de cada muestra en una goroutine
			go func(i int) {
				defer wg.Done()

				// Calcular la condición para actualizar (y * (X · W + b) >= 1)
				condition := y[i]*dotProduct(X[i], svm.Weights) + svm.Bias

				// Bloquear el acceso a los pesos y sesgo para evitar condiciones de carrera
				svm.mutex.Lock()
				defer svm.mutex.Unlock()

				if condition >= 1 {
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
			}(i)
		}

		// Esperar a que todas las goroutines de esta época finalicen
		wg.Wait()

		// Enviar progreso al cliente
		progress := float64(epoch+1) / float64(epochs) * 100
		conn.WriteJSON(map[string]interface{}{"progress": progress})
	}

	// Notificar al cliente que el entrenamiento se ha completado
	conn.WriteJSON(map[string]interface{}{"status": "Entrenamiento completado"})
}

// Función para predecir una muestra con el SVM entrenado y enviar progreso por WebSocket
func (svm *SVM) Predict(X [][]float64, conn *websocket.Conn) []float64 {
	predictions := make([]float64, len(X))
	nSamples := len(X)

	for i, sample := range X {
		if dotProduct(sample, svm.Weights)+svm.Bias >= 0 {
			predictions[i] = 1
		} else {
			predictions[i] = -1
		}

		// Enviar progreso de la predicción
		progress := float64(i+1) / float64(nSamples) * 100
		conn.WriteJSON(map[string]interface{}{"progress": progress})
	}

	// Notificar al cliente que la predicción se ha completado
	conn.WriteJSON(map[string]interface{}{"status": "Predicción completada"})
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
func AccuracyMetric_SVM(actual []float64, predicted []float64) float64 {
	correct := 0
	for i := range actual {
		if actual[i] == predicted[i] {
			correct++
		}
	}
	return (float64(correct) / float64(len(actual))) * 100.0
}
