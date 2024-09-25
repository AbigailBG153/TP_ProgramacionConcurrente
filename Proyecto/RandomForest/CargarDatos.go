package Random

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"sync"

	"github.com/gorilla/websocket"
)

// Estructura para pasar datos procesados
type DataBlock struct {
	Features [][]float64
	Labels   []float64
	Error    error
}

// Función para leer y procesar un archivo CSV en bloques concurrentemente con barra de progreso
func ReadCSVConcurrently(filepath string, blockSize int, conn *websocket.Conn) ([][]float64, []float64, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}

	// Buscar la posición de la columna 'CONSUMO'
	targetIndex := -1
	for i, header := range headers {
		if header == "CONSUMO" {
			targetIndex = i
			break
		}
	}

	if targetIndex == -1 {
		return nil, nil, fmt.Errorf("columna 'CONSUMO' no encontrada en el archivo CSV")
	}

	// Contar el total de filas para la barra de progreso
	var totalRows int
	for {
		_, err := reader.Read()
		if err != nil {
			break
		}
		totalRows++
	}

	// Resetear el reader y volver a leer desde el inicio
	file.Seek(0, 0)
	reader = csv.NewReader(file)
	_, _ = reader.Read() // Volver a leer los headers

	// Canales para los bloques de datos y errores
	dataChan := make(chan DataBlock)
	var wg sync.WaitGroup

	// Leer y procesar en bloques concurrentemente
	go func() {
		defer close(dataChan)
		block := make([][]string, 0, blockSize)

		for {
			row, err := reader.Read()
			if err != nil {
				if err.Error() == "EOF" {
					if len(block) > 0 {
						wg.Add(1)
						go ProcessBlock(block, targetIndex, dataChan, &wg)
					}
					break
				}
				dataChan <- DataBlock{Error: err}
				break
			}

			// Agregar la fila al bloque y enviar progreso al cliente
			block = append(block, row)
			progress := float64(len(block)) / float64(totalRows) * 100
			conn.WriteJSON(map[string]interface{}{"progress": progress})

			// Si el bloque alcanza el tamaño especificado, procesarlo
			if len(block) == blockSize {
				wg.Add(1)
				go ProcessBlock(block, targetIndex, dataChan, &wg)
				block = make([][]string, 0, blockSize) // Reiniciar el bloque
			}
		}
		wg.Wait()
	}()

	// Recolectar datos de los bloques procesados
	var allFeatures [][]float64
	var allLabels []float64

	for block := range dataChan {
		if block.Error != nil {
			return nil, nil, block.Error
		}
		allFeatures = append(allFeatures, block.Features...)
		allLabels = append(allLabels, block.Labels...)
	}

	return allFeatures, allLabels, nil
}

// Procesar un bloque de filas del CSV
func ProcessBlock(block [][]string, targetIndex int, dataChan chan<- DataBlock, wg *sync.WaitGroup) {
	defer wg.Done()
	var features [][]float64
	var labels []float64

	for _, row := range block {
		var featureRow []float64
		for i, value := range row {
			floatVal, err := strconv.ParseFloat(value, 64)
			if err != nil {
				dataChan <- DataBlock{Error: fmt.Errorf("error al convertir valor %v: %v", value, err)}
				return
			}
			if i == targetIndex {
				labels = append(labels, floatVal)
			} else {
				featureRow = append(featureRow, floatVal)
			}
		}
		features = append(features, featureRow)
	}
	dataChan <- DataBlock{Features: features, Labels: labels}
}

// Función para dividir los datos en entrenamiento y prueba con barra de progreso
func TrainTestSplit(data [][]float64, labels []float64, testSize float64, conn *websocket.Conn) ([][]float64, [][]float64, []float64, []float64) {
	n := len(data)
	testLen := int(float64(n) * testSize)
	indices := rand.Perm(n) // Permutación aleatoria de índices

	var X_train, X_test [][]float64
	var y_train, y_test []float64

	// Dividir los datos según los índices generados aleatoriamente y enviar progreso por WebSocket
	for i, idx := range indices {
		if i < testLen {
			X_test = append(X_test, data[idx])
			y_test = append(y_test, labels[idx])
		} else {
			X_train = append(X_train, data[idx])
			y_train = append(y_train, labels[idx])
		}

		// Enviar progreso al cliente mediante WebSocket
		progress := float64(i+1) / float64(n) * 100
		conn.WriteJSON(map[string]interface{}{"progress": progress})
	}

	// Notificar al cliente que la división de datos ha finalizado
	conn.WriteJSON(map[string]interface{}{"status": "División de datos completada"})
	return X_train, X_test, y_train, y_test
}

// Mapa para almacenar los datos cargados del CSV
var DatosEmpresa = make(map[string]map[string]map[string][]string)

// Lista para almacenar las empresas únicas
var ListaEmpresas []string

// Función para cargar datos desde el archivo CSV
func CargarDatosDesdeCSV(archivo string) error {
	file, err := os.Open(archivo)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return err
	}

	// Mapa auxiliar para evitar duplicados de empresas
	empresasUnicas := make(map[string]bool)

	// Recorrer los registros y llenar el mapa
	for _, record := range records[1:] { // Saltar la primera fila (cabecera)
		empresa := record[0]
		departamento := record[1]
		provincia := record[2]
		distrito := record[3]

		// Agregar empresa a la lista de empresas únicas
		if !empresasUnicas[empresa] {
			empresasUnicas[empresa] = true
			ListaEmpresas = append(ListaEmpresas, empresa)
		}

		// Inicializar los mapas si no existen
		if DatosEmpresa[empresa] == nil {
			DatosEmpresa[empresa] = make(map[string]map[string][]string)
		}
		if DatosEmpresa[empresa][departamento] == nil {
			DatosEmpresa[empresa][departamento] = make(map[string][]string)
		}
		// Añadir distrito a la provincia correspondiente
		DatosEmpresa[empresa][departamento][provincia] = append(DatosEmpresa[empresa][departamento][provincia], distrito)
	}

	return nil
}
