import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./App.css";
import { Box } from "@mui/material";
import Hero from "./Components/Hero/Hero";
import Programs from "./Components/Programs/Programs";
import Footer from "./Components/Footer/Footer";
import Join from "./Components/Join/Join";
import Exercises from "./Components/Exercises";
import SearchExercises from "./Components/SearchExercises";
import ExerciseDetail from "./Components/ExerciseDetails";
import { useState } from "react";
import Home from "./Components/Home";

function App() {
  const [exercises, setExercises] = useState([]);
  const [bodyPart, setBodyPart] = useState("all");

  return (
    <div className="App">
      {/* <Hero />
      <SearchExercises
        setExercises={setExercises}
        bodyPart={bodyPart}
        setBodyPart={setBodyPart}
      />
      <Exercises
        setExercises={setExercises}
        bodyPart={bodyPart}
        exercises={exercises}
      /> */}
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/exercise/:id" element={<ExerciseDetail />} />
          <Route path="/programs" element={<Programs />} />
          <Route path="/join" element={<Join />} />
          {/* <Route path="/exercise/:id" element={<ExerciseDetail />} /> */}
        </Routes>
      </BrowserRouter>
      <Footer />
    </div>
  );
}

export default App;
