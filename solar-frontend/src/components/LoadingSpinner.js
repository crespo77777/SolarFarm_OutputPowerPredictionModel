import React from 'react';
import './LoadingSpinner.css';

const LoadingSpinner = () => {
  return (
    <div className="spinner-container">
      <div className="loading-spinner"></div>
      <p>Učitavanje podataka...</p>
    </div>
  );
};

export default LoadingSpinner;
