package Random

import (
	"math"
	"math/rand"
	"sync"

	"github.com/gorilla/websocket"
)

// Estructura para representar un nodo del árbol
type Node struct {
	Left      *Node
	Right     *Node
	Feature   int
	Threshold float64
	Value     float64
	IsLeaf    bool
	X         [][]float64
	y         []float64
}

// Función para crear una muestra Bootstrap del dataset
func BootstrapSample(X [][]float64, y []float64) ([][]float64, []float64) {
	nSamples := len(X)
	sampleX := make([][]float64, nSamples)
	sampleY := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		idx := rand.Intn(nSamples)
		sampleX[i] = X[idx]
		sampleY[i] = y[idx]
	}
	return sampleX, sampleY
}

// Construir un árbol de decisión
func BuildTree(X [][]float64, y []float64, maxDepth int, minSize int) *Node {
	root := GetSplit(X, y)
	Split(root, maxDepth, minSize, 1)
	return root
}

// Función para dividir un nodo en base a los parámetros dados
func Split(node *Node, maxDepth, minSize, depth int) {
	leftX, leftY, rightX, rightY := SplitDataset(node.Feature, node.Threshold, node.X, node.y)
	if len(leftX) == 0 || len(rightX) == 0 {
		node.IsLeaf = true
		node.Value = ToTerminal(node.y)
		return
	}

	if depth >= maxDepth {
		node.Left = &Node{IsLeaf: true, Value: ToTerminal(leftY)}
		node.Right = &Node{IsLeaf: true, Value: ToTerminal(rightY)}
		return
	}

	if len(leftX) <= minSize {
		node.Left = &Node{IsLeaf: true, Value: ToTerminal(leftY)}
	} else {
		node.Left = GetSplit(leftX, leftY)
		Split(node.Left, maxDepth, minSize, depth+1)
	}

	if len(rightX) <= minSize {
		node.Right = &Node{IsLeaf: true, Value: ToTerminal(rightY)}
	} else {
		node.Right = GetSplit(rightX, rightY)
		Split(node.Right, maxDepth, minSize, depth+1)
	}
}

// Función para obtener la mejor división posible
func GetSplit(X [][]float64, y []float64) *Node {
	nFeatures := len(X[0])
	bestIndex, bestValue, bestScore := 0, 0.0, math.MaxFloat64

	for feature := 0; feature < nFeatures; feature++ {
		for _, row := range X {
			threshold := row[feature]
			_, leftY, _, rightY := SplitDataset(feature, threshold, X, y)
			groupsY := [][]float64{leftY, rightY}
			gini := GiniIndex(groupsY)
			if gini < bestScore {
				bestIndex = feature
				bestValue = threshold
				bestScore = gini
			}
		}
	}
	node := &Node{
		Feature:   bestIndex,
		Threshold: bestValue,
		X:         X,
		y:         y,
	}
	return node
}

// Función para dividir el dataset en base a un umbral y característica dada
func SplitDataset(feature int, threshold float64, X [][]float64, y []float64) (leftX [][]float64, leftY []float64, rightX [][]float64, rightY []float64) {
	for i, row := range X {
		if row[feature] < threshold {
			leftX = append(leftX, row)
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, row)
			rightY = append(rightY, y[i])
		}
	}
	return
}

// Calcular el índice Gini para una división
func GiniIndex(groups [][]float64) float64 {
	nInstances := 0.0
	for _, group := range groups {
		nInstances += float64(len(group))
	}
	gini := 0.0
	for _, group := range groups {
		size := float64(len(group))
		if size == 0 {
			continue
		}
		score := 0.0
		counts := ClassCounts(group)
		for _, count := range counts {
			p := count / size
			score += p * p
		}
		gini += (1.0 - score) * (size / nInstances)
	}
	return gini
}

// Contar las clases en un grupo
func ClassCounts(group []float64) map[float64]float64 {
	counts := make(map[float64]float64)
	for _, value := range group {
		counts[value]++
	}
	return counts
}

// Determinar el valor terminal de un nodo hoja
func ToTerminal(group []float64) float64 {
	counts := ClassCounts(group)
	var maxCount float64
	var classLabel float64
	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			classLabel = label
		}
	}
	return classLabel
}

// Función para predecir el resultado de una fila usando un árbol
func Predict(node *Node, row []float64) float64 {
	if node.IsLeaf {
		return node.Value
	}
	if row[node.Feature] < node.Threshold {
		return Predict(node.Left, row)
	} else {
		return Predict(node.Right, row)
	}
}

// Función para hacer predicciones usando un bosque de árboles de manera concurrente con progreso en WebSocket
func BaggingPredictConcurrent(forest []*Node, X [][]float64, conn *websocket.Conn) []float64 {
	predictions := make([]float64, len(X))
	var wg sync.WaitGroup
	resultChannel := make(chan struct {
		index      int
		prediction float64
	}, len(X))

	// Progreso inicial
	conn.WriteJSON(map[string]interface{}{"progress": 0})

	for i, row := range X {
		wg.Add(1)
		go func(i int, row []float64) {
			defer wg.Done()
			prediction := BaggingPredict(forest, row)
			resultChannel <- struct {
				index      int
				prediction float64
			}{i, prediction}

			// Enviar progreso al cliente
			progress := float64(i+1) / float64(len(X)) * 100
			conn.WriteJSON(map[string]interface{}{"progress": progress})
		}(i, row)
	}

	go func() {
		wg.Wait()
		close(resultChannel)
		conn.WriteJSON(map[string]interface{}{"status": "Predicción completada"})
	}()

	for result := range resultChannel {
		predictions[result.index] = result.prediction
	}

	return predictions
}

// Hacer una predicción usando el bosque de árboles
func BaggingPredict(forest []*Node, row []float64) float64 {
	predictions := make(map[float64]int)
	for _, tree := range forest {
		prediction := Predict(tree, row)
		predictions[prediction]++
	}
	var maxVotes int
	var finalPrediction float64
	for classValue, votes := range predictions {
		if votes > maxVotes {
			maxVotes = votes
			finalPrediction = classValue
		}
	}
	return finalPrediction
}

// Función para calcular la precisión
func AccuracyMetric_RF(actual []float64, predicted []float64) float64 {
	correct := 0
	for i := range actual {
		if actual[i] == predicted[i] {
			correct++
		}
	}
	return (float64(correct) / float64(len(actual))) * 100.0
}

// Construir un bosque aleatorio de árboles de decisión de manera concurrente y enviar progreso por WebSocket
func BuildRandomForestConcurrent(X_train_data [][]float64, y_train []float64, nTrees, maxDepth, minSize int, conn *websocket.Conn) []*Node {
	var wg sync.WaitGroup
	forest := make([]*Node, 0, nTrees)
	treeChannel := make(chan *Node, nTrees)
	sem := make(chan struct{}, 20)
	var mu sync.Mutex

	// Progreso inicial
	conn.WriteJSON(map[string]interface{}{"progress": 0})

	// Lanza las goroutines para construir los árboles
	for i := 0; i < nTrees; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			sampleX, sampleY := BootstrapSample(X_train_data, y_train)
			tree := BuildTree(sampleX, sampleY, maxDepth, minSize)

			// Bloquear acceso al canal
			mu.Lock()
			treeChannel <- tree
			progress := float64(len(forest)+1) / float64(nTrees) * 100
			conn.WriteJSON(map[string]interface{}{"progress": progress})
			mu.Unlock()
		}()
	}

	// Cerrar el canal cuando todas las goroutines terminen
	go func() {
		wg.Wait()
		close(treeChannel)
		conn.WriteJSON(map[string]interface{}{"status": "Entrenamiento completado"})
	}()

	// Recolectar los árboles del canal de manera eficiente
	for tree := range treeChannel {
		forest = append(forest, tree)
	}

	return forest
}
