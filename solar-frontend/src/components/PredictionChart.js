import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale, // Uključeno za potencijalnu upotrebu s vremenskim adapterima
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

const chartContainerStyle = {
  padding: "20px",
  margin: "20px auto", // Centriranje i margina
  border: "1px solid #ddd",
  borderRadius: "8px",
  backgroundColor: "#fff",
  maxWidth: "900px", // Ograničenje širine grafa
};

const PredictionChart = ({ predictionData, period }) => {
  if (!predictionData) {
    // Provjera samo za null, prazan niz se obrađuje ispod
    return null; // Nemoj ništa prikazivati ako je predictionData null (npr. inicijalno stanje)
  }
  if (predictionData.length === 0) {
    return (
      <div style={chartContainerStyle}>
        <p style={{ textAlign: "center", color: "#777" }}>
          Nema dostupnih podataka za prikaz za odabrani period.
        </p>
      </div>
    );
  }

  const chartData = {
    labels: predictionData.map((data) =>
      new Date(data.timestamp).toLocaleString("hr-HR", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    ),
    datasets: [
      {
        label: `Predviđena solarna snaga (V) - ${period
          .replace("h", " sata")
          .replace("d", " dana")}`,
        data: predictionData.map((data) => data.predicted_solar_power_kw),
        fill: true, // Može biti true za ispunu ispod linije
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        borderColor: "rgb(75, 192, 192)",
        tension: 0.1,
        pointRadius: period === "24h" ? 3 : 2, // Manje točke za duže periode
        pointHoverRadius: period === "24h" ? 5 : 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true, // Održava omjer širine i visine definiran u canvasu ili postavljen stilom
    plugins: {
      legend: {
        position: "top",
        labels: {
          font: {
            size: 14,
          },
        },
      },
      title: {
        display: true,
        text: "Grafički prikaz generiranog napona iz solarne energije",
        font: {
          size: 18,
          weight: "bold",
        },
        padding: {
          top: 10,
          bottom: 20,
        },
      },
      tooltip: {
        mode: "index",
        intersect: false,
        callbacks: {
          label: function (context) {
            let label = context.dataset.label || "";
            if (label) {
              label += ": ";
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y.toFixed(2) + " V";
            }
            return label;
          },
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Vrijeme",
          font: {
            size: 14,
            weight: "bold",
          },
        },
        ticks: {
          autoSkip: true,
          maxTicksLimit: period === "24h" ? 12 : period === "72h" ? 10 : 7, // Prilagodi broj oznaka
          font: {
            size: 10,
          },
        },
      },
      y: {
        title: {
          display: true,
          text: "Predviđen napon (V)",
          font: {
            size: 14,
            weight: "bold",
          },
        },
        beginAtZero: true,
        ticks: {
          font: {
            size: 10,
          },
        },
      },
    },
  };

  return (
    <div style={chartContainerStyle}>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default PredictionChart;
