import React from "react";

const Header = () => {
  const headerStyle = {
    backgroundColor: "#2c3e50", // Tamno plavo-siva boja
    color: "white",
    padding: "1rem",
    textAlign: "center",
    fontSize: "1.5rem",
    marginBottom: "20px",
    height: "10vh",
  };
  return (
    <header style={headerStyle}>
      <h1 style={{ margin: "1vh" }}>
        Predikcija generiranog napona iz solarne energije
      </h1>
    </header>
  );
};

export default Header;
