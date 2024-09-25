package Random

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"
	"time"
)

// Estructura para capturar los datos recibidos del formulario
type FormData struct {
	Empresa           string
	Departamento      string
	Provincia         string
	Distrito          string
	FechaConsumoDesde string
	FechaConsumoHasta string
	Importe           float64
	Consumo           float64
	Tarifa            string
}

// Estructura para capturar la data transformada
type TransformedData struct {
	DiasConsumo       int
	MesConsumo        int
	FechaConsumoDesde string
	FechaConsumoHasta string
	Importe           float64
	Consumo           float64
	Tarifa            string
	Departamento      string
	Provincia         string
	Distrito          string
}

// Función que transforma los datos crudos recibidos del formulario
func TransformarData(data FormData) (TransformedData, error) {
	// Convertir fechas a tipo time.Time
	fechaDesde, err := time.Parse("2006-01-02", data.FechaConsumoDesde)
	if err != nil {
		return TransformedData{}, fmt.Errorf("error al convertir FECHA_CONSUMO_DESDE: %v", err)
	}
	fechaHasta, err := time.Parse("2006-01-02", data.FechaConsumoHasta)
	if err != nil {
		return TransformedData{}, fmt.Errorf("error al convertir FECHA_CONSUMO_HASTA: %v", err)
	}

	// Calcular la duración del consumo en días
	diasConsumo := int(fechaHasta.Sub(fechaDesde).Hours() / 24)
	// Obtener el mes del consumo
	mesConsumo := fechaHasta.Month()

	return TransformedData{
		DiasConsumo:       diasConsumo,
		MesConsumo:        int(mesConsumo),
		FechaConsumoDesde: data.FechaConsumoDesde,
		FechaConsumoHasta: data.FechaConsumoHasta,
		Importe:           data.Importe,
		Consumo:           data.Consumo,
		Tarifa:            data.Tarifa,
		Departamento:      data.Departamento,
		Provincia:         data.Provincia,
		Distrito:          data.Distrito,
	}, nil
}

// Función para cargar las cabeceras del archivo CSV de entrenamiento
func CargarCabeceras(archivo string) ([]string, error) {
	file, err := os.Open(archivo)
	if err != nil {
		return nil, fmt.Errorf("error al abrir el archivo CSV: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	// Leer solo la primera fila (las cabeceras)
	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("error al leer las cabeceras del CSV: %v", err)
	}

	return headers, nil
}

// Ajustar los datos transformados a la estructura de las cabeceras del modelo entrenado
func AjustarData(transformedData TransformedData, headers []string) []float64 {
	// Mapa para almacenar los datos transformados con sus nombres de columnas
	dataMap := make(map[string]float64)

	// Asignar los valores de fechas en el mapa usando nanosegundos desde la época
	dataMap["FECHA_COSNUMO_DESDE"] = parseFechaToNano(transformedData.FechaConsumoDesde)
	dataMap["FECHA_CONSUMO_HASTA"] = parseFechaToNano(transformedData.FechaConsumoHasta)

	// Asignar otros valores numéricos
	dataMap["DIAS_CONSUMO"] = float64(transformedData.DiasConsumo)
	dataMap["MES_CONSUMO"] = float64(transformedData.MesConsumo)
	dataMap["IMPORTE"] = transformedData.Importe
	dataMap["CONSUMO"] = transformedData.Consumo

	// Buscar y asignar los valores de los campos dinámicos (tarifas, departamentos, provincias, distritos)
	for _, header := range headers {
		if strings.EqualFold(header, "TARIFA_"+transformedData.Tarifa) {
			dataMap[header] = 1
		} else if strings.EqualFold(header, "DEPARTAMENTO_"+transformedData.Departamento) {
			dataMap[header] = 1
		} else if strings.EqualFold(header, "PROVINCIA_"+transformedData.Provincia) {
			dataMap[header] = 1
		} else if strings.EqualFold(header, "DISTRITO_"+transformedData.Distrito) {
			dataMap[header] = 1
		} else {
			// Mantener cualquier otro valor como 0 si no se reconoce específicamente
			if _, exists := dataMap[header]; !exists {
				dataMap[header] = 0
			}
		}
	}

	// Crear una lista ordenada de valores de acuerdo con las cabeceras del modelo
	result := make([]float64, len(headers))
	for i, header := range headers {
		result[i] = dataMap[header]
	}

	return result
}

// Convertir la fecha de formato string a nanosegundos desde la época
func parseFechaToNano(fecha string) float64 {
	t, err := time.Parse("2006-01-02", fecha)
	if err != nil {
		return 0 // En caso de error, se devuelve 0
	}
	// Convertir a nanosegundos desde la época
	return float64(t.UnixNano())
}

// Función para guardar los datos ajustados en un nuevo CSV
func GuardarDatosAjustados(data []float64, headers []string, archivo string) error {
	file, err := os.Create(archivo)
	if err != nil {
		return fmt.Errorf("error al crear el archivo CSV: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Escribir las cabeceras en el nuevo CSV
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("error al escribir las cabeceras: %v", err)
	}

	// Convertir los datos ajustados a string para escribirlos en el CSV
	row := make([]string, len(data))
	for i, val := range data {
		row[i] = fmt.Sprintf("%.0f", val) // Formato de número a string sin decimales
	}

	// Escribir la fila de datos ajustados
	if err := writer.Write(row); err != nil {
		return fmt.Errorf("error al escribir los datos ajustados: %v", err)
	}

	return nil
}
