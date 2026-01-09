import { React, useState } from "react";
import { Box } from "@mui/material";

import Exercises from "./Exercises";
import SearchExercises from "./SearchExercises";
import Hero from "./Hero/Hero";

const Home = () => {
  const [exercises, setExercises] = useState([]);
  const [bodyPart, setBodyPart] = useState("all");
  return (
    <div>
      <Hero />
      <SearchExercises
        setExercises={setExercises}
        bodyPart={bodyPart}
        setBodyPart={setBodyPart}
      />
      <Exercises
        setExercises={setExercises}
        bodyPart={bodyPart}
        exercises={exercises}
      />
    </div>
  );
};

export default Home;
