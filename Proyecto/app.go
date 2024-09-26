package main

import (
	"Proyecto/Modelado"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Permitir todas las conexiones
	},
}

// Variables globales
var (
	X_train, X_test [][]float64
	y_train, y_test []float64
	labels          []float64
	data            [][]float64
	svm             Modelado.SVM
	forest          []*Modelado.Node
	algorithm       string // Variable para guardar el algoritmo seleccionado
)

func main() {
	// Crear una nueva instancia de Gin
	r := gin.Default()

	// Cargar las plantillas HTML
	r.SetFuncMap(template.FuncMap{
		"safe": func(str string) template.HTML {
			return template.HTML(str)
		},
	})
	r.LoadHTMLGlob("templates/*")

	// Ruta principal para mostrar el formulario
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	// Ruta para manejar la carga de archivos
	r.POST("/upload", handleFileUpload)

	// Ruta WebSocket para cargar datos
	r.GET("/ws", func(c *gin.Context) {
		handleLoadData(c.Writer, c.Request)
	})

	// Ruta WebSocket para particionar datos
	r.GET("/partition", func(c *gin.Context) {
		handlePartitionData(c.Writer, c.Request)
	})

	// Ruta para entrenar el modelo
	r.GET("/train", func(c *gin.Context) {
		handleFit(c.Writer, c.Request)
	})

	// Ruta para predecir con el modelo entrenado
	r.GET("/prediction", func(c *gin.Context) {
		handlePrediction(c.Writer, c.Request)
	})

	log.Println("Servidor ejecutándose en :8080")
	r.Run(":8080") // Ejecuta el servidor en el puerto 8080
}

func handleFileUpload(c *gin.Context) {
	// Obtener el archivo del formulario
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No se pudo obtener el archivo"})
		return
	}

	// Crear el directorio de uploads si no existe
	os.MkdirAll("uploads", os.ModePerm)

	// Guardar el archivo subido en una ruta temporal
	savePath := filepath.Join("uploads", filepath.Base(file.Filename))
	if err := c.SaveUploadedFile(file, savePath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "No se pudo guardar el archivo"})
		return
	}

	// Enviar la ruta del archivo como respuesta
	c.JSON(http.StatusOK, gin.H{"status": "Archivo subido correctamente", "filepath": savePath})
}

func handleLoadData(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Error al establecer la conexión WebSocket:", err)
		return
	}
	defer conn.Close()

	// Obtener la ruta del archivo desde la URL del WebSocket
	filepath := r.URL.Query().Get("filepath")
	if filepath == "" {
		log.Println("Ruta del archivo no proporcionada")
		return
	}

	blockSize := 1000 // Ajusta este valor según tus necesidades

	fmt.Println("Iniciando carga de datos...")
	startTime := time.Now()
	data, labels, err = Modelado.ReadCSVConcurrently(filepath, blockSize, conn)
	if err != nil {
		log.Printf("Error al leer el CSV: %v", err)
		conn.WriteJSON(map[string]interface{}{"error": "Error al procesar el archivo"})
		return
	}
	loadDataTime := time.Since(startTime)
	fmt.Printf("\nCarga de datos completada en: %s\n", loadDataTime)
	fmt.Printf("Data cargada: %d filas\n", len(data))
	fmt.Printf("Labels cargados: %d filas\n", len(labels))

	// Notificar al cliente que la carga se ha completado
	conn.WriteJSON(map[string]interface{}{"status": "Carga completada"})
}

func handlePartitionData(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Error al establecer la conexión WebSocket:", err)
		return
	}
	defer conn.Close()

	// Obtener el tamaño del conjunto de prueba desde la URL del WebSocket
	testSizeStr := r.URL.Query().Get("testSize")
	testSize, err := strconv.ParseFloat(testSizeStr, 64)
	if err != nil || testSize <= 0 || testSize >= 1 {
		conn.WriteJSON(map[string]interface{}{"error": "Valor de testSize inválido"})
		return
	}

	// Ajustar etiquetas solo si se usa SVM
	switch algorithm {
	case "svm":
		// Convertir etiquetas para SVM (se espera -1 o 1)
		for i := 0; i < len(labels); i++ {
			if labels[i] == 0 {
				labels[i] = -1
			} else {
				labels[i] = 1
			}
		}
		fmt.Println("Etiquetas ajustadas para SVM")

	case "random_forest":
		fmt.Println("Etiquetas mantenidas para Random Forest")

	default:
		conn.WriteJSON(map[string]interface{}{"error": "Algoritmo no soportado para partición de datos"})
		return
	}

	// Iniciar la separación de datos usando el testSize proporcionado por el usuario
	fmt.Println("Iniciando separación de datos...")
	startTime := time.Now()
	X_train, X_test, y_train, y_test = Modelado.TrainTestSplit(data, labels, testSize, conn)
	separationTime := time.Since(startTime)
	fmt.Printf("\nSeparación de datos completada en: %s\n", separationTime)
	fmt.Printf("X_train: %d\n", len(X_train))
	fmt.Printf("X_test: %d\n", len(X_test))
	fmt.Printf("y_train: %d\n", len(y_train))
	fmt.Printf("y_test: %d\n", len(y_test))

	// Notificar al cliente que la partición se ha completado
	conn.WriteJSON(map[string]interface{}{"status": "Partición completada"})
}

func handleFit(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Error al establecer la conexión WebSocket:", err)
		return
	}
	defer conn.Close()

	// Obtener los parámetros desde la URL del WebSocket
	algorithm = r.URL.Query().Get("algorithm") // Obtener el algoritmo seleccionado
	learningRate, err := strconv.ParseFloat(r.URL.Query().Get("learningRate"), 64)
	if err != nil {
		conn.WriteJSON(map[string]interface{}{"error": "Valor de learningRate inválido"})
		return
	}

	lambda, err := strconv.ParseFloat(r.URL.Query().Get("lambda"), 64)
	if err != nil {
		conn.WriteJSON(map[string]interface{}{"error": "Valor de lambda inválido"})
		return
	}

	epochs, err := strconv.Atoi(r.URL.Query().Get("epochs"))
	if err != nil {
		conn.WriteJSON(map[string]interface{}{"error": "Valor de epochs inválido"})
		return
	}

	// Entrenamiento según el algoritmo seleccionado
	switch algorithm {
	case "svm":
		// Definir hiperparámetros del SVM
		svm = Modelado.SVM{
			LearningRate: learningRate,
			Lambda:       lambda,
		}

		fmt.Println("Iniciando Entrenamiento SVM...")
		startTrain := time.Now()
		svm.Train(X_train, y_train, epochs, conn)
		trainDuration := time.Since(startTrain)
		fmt.Printf("Tiempo de entrenamiento SVM: %v\n", trainDuration)

	case "random_forest":
		// Obtener parámetros específicos de Random Forest
		nTrees, err := strconv.Atoi(r.URL.Query().Get("nTrees"))
		if err != nil || nTrees < 1 {
			conn.WriteJSON(map[string]interface{}{"error": "Número de árboles inválido"})
			return
		}

		maxDepth, err := strconv.Atoi(r.URL.Query().Get("maxDepth"))
		if err != nil || maxDepth < 1 {
			conn.WriteJSON(map[string]interface{}{"error": "Profundidad máxima inválida"})
			return
		}

		minSize, err := strconv.Atoi(r.URL.Query().Get("minSize"))
		if err != nil || minSize < 1 {
			conn.WriteJSON(map[string]interface{}{"error": "Tamaño mínimo inválido"})
			return
		}

		fmt.Println("Iniciando Entrenamiento Random Forest...")
		startTrain := time.Now()
		forest = Modelado.BuildRandomForestConcurrent(X_train, y_train, nTrees, maxDepth, minSize, conn)
		trainDuration := time.Since(startTrain)
		fmt.Printf("Tiempo de entrenamiento Random Forest: %v\n", trainDuration)

	default:
		conn.WriteJSON(map[string]interface{}{"error": "Algoritmo no soportado"})
		return
	}

	// Notificar al cliente que el entrenamiento se ha completado
	conn.WriteJSON(map[string]interface{}{
		"status": "Entrenamiento completado",
	})
}

func handlePrediction(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Error al establecer la conexión WebSocket:", err)
		return
	}
	defer conn.Close()

	var predictions []float64
	var predictDuration time.Duration
	var accuracy float64

	// Predicción según el algoritmo seleccionado
	switch algorithm {
	case "svm":
		fmt.Println("Iniciando Predicción con SVM...")
		startPredict := time.Now()
		predictions = svm.Predict(X_test, conn)
		predictDuration = time.Since(startPredict)
		fmt.Printf("Tiempo de predicción SVM: %v\n", predictDuration)

		accuracy = Modelado.AccuracyMetric_SVM(y_test, predictions)

	case "random_forest":
		fmt.Println("Iniciando Predicción con Random Forest...")
		startPredict := time.Now()
		predictions = Modelado.BaggingPredictConcurrent(forest, X_test, conn)
		predictDuration = time.Since(startPredict)
		fmt.Printf("Tiempo de predicción Random Forest: %v\n", predictDuration)

		accuracy = Modelado.AccuracyMetric_RF(y_test, predictions)

	default:
		conn.WriteJSON(map[string]interface{}{"error": "Algoritmo no soportado para predicción"})
		return
	}

	fmt.Printf("Precisión: %.2f%%\n", accuracy)

	// Enviar resultados al cliente
	conn.WriteJSON(map[string]interface{}{
		"status":      "Predicción completada",
		"accuracy":    accuracy,
		"predictTime": predictDuration.String(),
	})
}
