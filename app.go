package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"proyectoGo2/Random"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

const entrenamientoCSV = "../Data/datos_procesados_1000.csv"
const ajustadosCSV = "../Data/data_ingresada.csv"

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
	svm             Random.SVM
	forest          []*Random.Node
	algorithm       string // Variable para guardar el algoritmo seleccionado
)

func main() {

	// Cargar los datos desde el CSV
	err := Random.CargarDatosDesdeCSV("./Data/empresas_ubicaciones.csv") // Cambia a la ruta correcta de tu archivo CSV
	if err != nil {
		log.Fatalf("Error al cargar datos desde el CSV: %v", err)
	}
	// Cargar las cabeceras del CSV de entrenamiento
	headers, err := Random.CargarCabeceras(entrenamientoCSV)
	if err != nil {
		log.Fatalf("Error al cargar las cabeceras del CSV de entrenamiento: %v", err)
	}

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

	// Ruta para obtener la lista de empresas
	r.GET("/api/empresas", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"empresas": Random.ListaEmpresas})
	})

	// Ruta para obtener departamentos de una empresa
	r.GET("/api/departamentos", func(c *gin.Context) {
		empresa := c.Query("empresa")
		departamentos, exists := Random.DatosEmpresa[empresa]
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Empresa no encontrada"})
			return
		}

		departamentoList := make([]string, 0, len(departamentos))
		for departamento := range departamentos {
			departamentoList = append(departamentoList, departamento)
		}

		c.JSON(http.StatusOK, gin.H{"departamentos": departamentoList})
	})

	// Ruta para obtener provincias de un departamento
	r.GET("/api/provincias", func(c *gin.Context) {
		empresa := c.Query("empresa")
		departamento := c.Query("departamento")
		empresas, exists := Random.DatosEmpresa[empresa]
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Empresa no encontrada"})
			return
		}
		provincias, exists := empresas[departamento]
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Departamento no encontrado"})
			return
		}

		provinciaList := make([]string, 0, len(provincias))
		for provincia := range provincias {
			provinciaList = append(provinciaList, provincia)
		}

		c.JSON(http.StatusOK, gin.H{"provincias": provinciaList})
	})

	// Ruta para obtener distritos de una provincia
	r.GET("/api/distritos", func(c *gin.Context) {
		empresa := c.Query("empresa")
		departamento := c.Query("departamento")
		provincia := c.Query("provincia")

		empresas, exists := Random.DatosEmpresa[empresa]
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Empresa no encontrada"})
			return
		}
		provincias, exists := empresas[departamento]
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Departamento no encontrado"})
			return
		}
		distritos, exists := provincias[provincia]
		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "Provincia no encontrada"})
			return
		}

		c.JSON(http.StatusOK, gin.H{"distritos": distritos})
	})

	// Ruta para manejar el envío del formulario
	r.POST("/submit", func(c *gin.Context) {
		var data Random.FormData

		// Captura los datos del formulario
		data.Empresa = c.PostForm("empresa")
		data.Departamento = c.PostForm("departamento")
		data.Provincia = c.PostForm("provincia")
		data.Distrito = c.PostForm("distrito")
		data.FechaConsumoDesde = c.PostForm("fecha_consumo_desde")
		data.FechaConsumoHasta = c.PostForm("fecha_consumo_hasta")
		data.Importe, _ = strconv.ParseFloat(c.PostForm("importe"), 64)
		data.Consumo, _ = strconv.ParseFloat(c.PostForm("consumo"), 64)
		data.Tarifa = c.PostForm("tarifa")

		// Mostrar log de los datos capturados del formulario
		log.Printf("Datos capturados del formulario: %+v", data)

		// Transformar los datos del formulario
		transformedData, err := Random.TransformarData(data)
		if err != nil {
			log.Printf("Error transformando datos: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Error transformando datos"})
			return
		}

		// Ajustar los datos a la estructura de las cabeceras
		ajustados := Random.AjustarData(transformedData, headers)

		// Mostrar log de los datos ajustados
		log.Printf("Datos ajustados: %v", ajustados)

		// Guardar los datos ajustados en un nuevo CSV
		err = Random.GuardarDatosAjustados(ajustados, headers, ajustadosCSV)
		if err != nil {
			log.Printf("Error guardando los datos ajustados: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Error guardando los datos ajustados"})
			return
		}

		// Confirmación de éxito
		c.JSON(http.StatusOK, gin.H{"status": "success"})
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
	data, labels, err = Random.ReadCSVConcurrently(filepath, blockSize, conn)
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
	log.Printf("Tamaño del conjunto de prueba recibido: %s\n", testSizeStr) // Log para verificar el valor
	algorithm = r.URL.Query().Get("algorithm")
	testSize, err := strconv.ParseFloat(testSizeStr, 64)
	if err != nil || testSize <= 0 || testSize >= 1 {
		conn.WriteJSON(map[string]interface{}{"error": "Valor de testSize inválido"})
		return
	}

	if algorithm == "svm" {
		for i := 0; i < len(labels); i++ {
			if labels[i] == 0 {
				labels[i] = -1
			} else {
				labels[i] = 1
			}
		}
	}

	// Iniciar la separación de datos usando el testSize proporcionado por el usuario
	fmt.Println("Iniciando separación de datos...")
	startTime := time.Now()
	X_train, X_test, y_train, y_test = Random.TrainTestSplit(data, labels, testSize, conn)
	separationTime := time.Since(startTime)
	fmt.Printf("\nSeparación de datos completada en: %s\n", separationTime)
	fmt.Printf("X_train: %d\n", len(X_train))
	fmt.Printf("X_test: %d\n", len(X_test))
	fmt.Printf("y_train: %d\n", len(y_train))
	fmt.Printf("y_test: %d\n", len(y_test))

	// Enviar mensaje de partición completada
	err = conn.WriteJSON(map[string]interface{}{"status": "Partición completada"})
	if err != nil {
		log.Println("Error al enviar mensaje de completación:", err)
	} else {
		log.Println("Partición completada y mensaje enviado.")
	}
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
		svm = Random.SVM{
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
		forest = Random.BuildRandomForestConcurrent(X_train, y_train, nTrees, maxDepth, minSize, conn)
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
		accuracy = Random.AccuracyMetric_SVM(y_test, predictions)
		fmt.Printf("Precisión: %.2f%%\n", accuracy)

		// Crear la respuesta
		response := map[string]interface{}{
			"status":   "Entrenamiento completado accuracy",
			"accuracy": accuracy,
		}

		// Establecer el tipo de contenido y enviar la respuesta
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)

	case "random_forest":
		fmt.Println("Iniciando Predicción con Random Forest...")
		startPredict := time.Now()
		predictions = Random.BaggingPredictConcurrent(forest, X_test, conn)
		predictDuration = time.Since(startPredict)
		fmt.Printf("Tiempo de predicción Random Forest: %v\n", predictDuration)
		fmt.Printf("Precisión: %.2f%%\n", accuracy)
		accuracy = Random.AccuracyMetric_RF(y_test, predictions)

	default:
		conn.WriteJSON(map[string]interface{}{"error": "Algoritmo no soportado para predicción"})
		return
	}

}
