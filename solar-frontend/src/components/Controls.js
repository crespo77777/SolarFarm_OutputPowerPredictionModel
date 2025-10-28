import React, { useState } from 'react';

const controlStyle = {
  padding: '20px',
  textAlign: 'center',
  border: '1px solid #eee',
  margin: '0 auto 20px auto', // Centriranje i margina
  borderRadius: '8px',
  backgroundColor: '#f9f9f9',
  maxWidth: '600px', // Ograničenje širine radi boljeg izgleda
};

const selectStyle = {
  padding: '10px 12px',
  marginRight: '15px',
  fontSize: '1rem',
  borderRadius: '4px',
  border: '1px solid #ccc',
  minWidth: '200px',
};

const buttonStyle = {
  padding: '10px 25px',
  fontSize: '1rem',
  color: 'white',
  backgroundColor: '#3498db', // Plava boja
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
  transition: 'background-color 0.2s ease-in-out',
};

const buttonHoverStyle = {
  backgroundColor: '#2980b9', // Tamnija plava za hover
};

const Controls = ({ onFetchPrediction, isLoading }) => {
  const [selectedPeriod, setSelectedPeriod] = useState('24h');
  const [isButtonHovered, setIsButtonHovered] = useState(false);

  const periodOptions = [
    { label: '24 sata', value: '24h' },
    { label: '3 dana (72 sata)', value: '72h' },
    { label: '7 dana', value: '7d' },
  ];

  const handleSubmit = () => {
    onFetchPrediction(selectedPeriod);
  };

  const currentButtonStyle = isLoading 
    ? { ...buttonStyle, backgroundColor: '#a5c9e3', cursor: 'not-allowed' } 
    : isButtonHovered 
      ? { ...buttonStyle, ...buttonHoverStyle }
      : buttonStyle;

  return (
    <div style={controlStyle}>
      <label htmlFor="period-select" style={{ marginRight: '10px', fontSize: '1.1rem', display: 'block', marginBottom: '10px' }}>
        Odaberite period prognoze:
      </label>
      <div>
        <select
          id="period-select"
          value={selectedPeriod}
          onChange={(e) => setSelectedPeriod(e.target.value)}
          style={selectStyle}
          disabled={isLoading}
        >
          {periodOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <button 
          onClick={handleSubmit} 
          style={currentButtonStyle}
          disabled={isLoading}
          onMouseEnter={() => setIsButtonHovered(true)}
          onMouseLeave={() => setIsButtonHovered(false)}
        >
          {isLoading ? 'Dohvaćam...' : 'Dohvati Prognozu i Predikciju'}
        </button>
      </div>
    </div>
  );
};

export default Controls;
