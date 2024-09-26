package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"proyectoGo/modelo"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func handleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("Error al crear WebSocket: %v", err)
		return
	}
	defer conn.Close()

	for {
		// Recibir datos desde WebSocket
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Printf("Error al leer mensaje: %v", err)
			break
		}

		// Parsear el mensaje JSON
		var inputData map[string]string
		err = json.Unmarshal(msg, &inputData)
		if err != nil {
			log.Printf("Error al parsear JSON: %v", err)
			break
		}

		// Obtener los parámetros del SVM
		alpha, _ := strconv.ParseFloat(inputData["alpha"], 64)
		epochs, _ := strconv.Atoi(inputData["epochs"])
		dataX := inputData["dataX"]
		dataY := inputData["dataY"]

		// Cargar características y etiquetas desde los strings recibidos
		X := modelo.ParseCSVData(dataX)
		y := modelo.ParseCSVLabels(dataY)

		// Crear y entrenar el modelo
		svm := modelo.SVM{
			Alpha:  alpha,
			Epochs: epochs,
		}
		trainSet, testSet := modelo.SplitData(X, y, 0.7)
		svm.Train(trainSet)

		// Evaluar el modelo
		result := svm.Evaluate(testSet)

		// Enviar resultado al cliente
		response := fmt.Sprintf("Entrenamiento completado. Precisión: %.2f%%", result*100)
		conn.WriteMessage(websocket.TextMessage, []byte(response))
	}
}

func main() {
	// Crear una nueva instancia de Gin
	r := gin.Default()

	// Configurar la carpeta para las plantillas HTML
	r.LoadHTMLGlob("templates/*")

	// Ruta principal que renderiza el HTML
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", gin.H{
			"title": "SVM Training",
		})
	})

	// Ruta para WebSocket
	r.GET("/ws", handleWebSocket)

	r.Run(":8080")
}
