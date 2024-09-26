package modelo

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

// Datos de entrenamiento: características y etiquetas
type DataPoint struct {
	features []float64
	label    float64
}

// Función para cargar el archivo X.csv (ignorando la primera fila de cabeceras)
func LoadFeatures(filename string) ([][]float64, error) {
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
		PrintProgressBar(i+1, len(records), "Cargando características")
	}
	fmt.Printf("\nTiempo de carga de características: %v\n", time.Since(start))
	return features, nil
}

// Función para cargar el archivo y.csv (ignorando la primera fila de cabeceras)
func LoadLabels(filename string) ([]float64, error) {
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
		PrintProgressBar(i+1, len(records), "Cargando etiquetas")
	}
	fmt.Printf("\nTiempo de carga de etiquetas: %v\n", time.Since(start))
	return labels, nil
}

func ParseCSVData(data string) [][]float64 {
	reader := csv.NewReader(strings.NewReader(data))
	rawData, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Error al parsear datos CSV: %v", err)
	}
	var result [][]float64
	for _, row := range rawData {
		var rowData []float64
		for _, val := range row {
			num, _ := strconv.ParseFloat(val, 64)
			rowData = append(rowData, num)
		}
		result = append(result, rowData)
	}
	return result
}

func ParseCSVLabels(data string) []float64 {
	reader := csv.NewReader(strings.NewReader(data))
	rawData, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Error al parsear etiquetas CSV: %v", err)
	}
	var result []float64
	for _, row := range rawData {
		for _, val := range row {
			num, _ := strconv.ParseFloat(val, 64)
			result = append(result, num)
		}
	}
	return result
}
