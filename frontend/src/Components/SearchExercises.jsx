import React, { useEffect, useState } from "react";
import { Box, Button, Stack, TextField, Typography } from "@mui/material";

import { exerciseOptions, fetchData } from "../utils/fetchData";
import HorizontalScrollbar from "./HorizontalScrollbar";

const SearchExercises = ({ setExercises, bodyPart, setBodyPart }) => {
  const [search, setSearch] = useState("");
  const [bodyParts, setBodyParts] = useState([]);

  useEffect(() => {
    const fetchExercisesData = async () => {
      const bodyPartsData = await fetchData(
        "https://exercisedb.p.rapidapi.com/exercises/bodyPartList",
        exerciseOptions
      );

      setBodyParts(["all", ...bodyPartsData]);
    };

    fetchExercisesData();
  }, []);

  const handleSearch = async () => {
    if (search) {
      const exercisesData = await fetchData(
        "https://exercisedb.p.rapidapi.com/exercises?limit=1327",
        exerciseOptions
      );
      console.log(exercisesData);

      // const handleSearch = async () => {
      //   if (search) {
      //     const selectedExercise = [
      //       "Burpee",
      //       "Jump rope",
      //       "Mountain climber",
      //       "Ski step",
      //       // Add other exercises you want to display here
      //     ];

      //     window.scrollTo({ top: 1800, left: 100, behavior: "smooth" });

      //     setSearch("");
      //     setExercises(selectedExercise);
      //   }
      // };

      const selectedExercise = [
        "burpee",
        "jump rope",
        "mountain climber",
        "ski step",
        "dynamic chest stretch(male)",
        "push-up",
        "raise single arm push-up",
        "wide hand push up",
        "resistance band seated chest press",
        "neck side stretch",
        "side push neck stretch",
        "ankle circles",
        "bodyweight standing calf raise",
        "circles knee stretch",
        "one leg floor calf raise",
        "dumbbell seated palms up wrist curl",
        "side wrist pull stretch",
        "wrist circles",
        "jump squat",
        "Squat to overhead reach",
        "split squat",
        "dumbbell lunge",
        "forward lunge(male)",
        "cross body crunch",
        "jackknife situp",
        "prisoner half sit-up(male)",
        "russian twist",
        "shoulder tap",
        "sit-up with arms on chest",
        "dumbbell alternate biceps curl",
        "dumbbell biceps curl squat",
        "dumbbell cross body hammer curl",
        "dumbbell high curl",
        "dumbbell one arm standing curl",
        "dumbbell waiter biceps curl",
        "dumbbell alternate side press",
        "dumbbell front raise",
        "dumbbell lateral raise",
        "dumbbell one arm shoulder press",
        "dumbbell standing overhead press",
        "shoulder tap",
        "dumbbell upright row",
        "dumbbell bent over",
        "dumbbell shrug",
        "lower back curl",
        "sphinx",
        "air bike",
        "alternate heel touchers",
        "standing lateral stretch",
        "upper back stretch",
        "upward facing dog",
      ];

      const searchedExercises = exercisesData.filter((item) =>
        selectedExercise.includes(item.name.toLowerCase())
      );
      const searchedExercisesModified = searchedExercises.filter(
        (item) =>
          item.name.toLowerCase().includes(search) ||
          item.target.toLowerCase().includes(search) ||
          item.equipment.toLowerCase().includes(search) ||
          item.bodyPart.toLowerCase().includes(search)
      );
      console.log(searchedExercisesModified);

      window.scrollTo({ top: 1800, left: 100, behavior: "smooth" });

      setSearch("");
      setExercises(searchedExercisesModified);
    }
  };
  // const handleSearch = async () => {
  //   if (search) {
  //     try {
  //       const exercisesData = await fetchData(
  //         "https://exercisedb.p.rapidapi.com/exercises?limit=1327",
  //         exerciseOptions
  //       );

  //       // Uploading exercisesData to a new file named data.json
  //       const uploadData = async (data) => {
  //         try {
  //           const jsonData = JSON.stringify(data);
  //           const blob = new Blob([jsonData], { type: "application/json" });
  //           const formData = new FormData();
  //           formData.append("file", blob, "data.json");

  //           const response = await fetch(
  //             "http://localhost:3000/upload", // Correct endpoint
  //             {
  //               method: "POST",
  //               body: formData,
  //             }
  //           );
  //           console.log("Upload response:", response);
  //         } catch (error) {
  //           console.error("Error uploading data:", error);
  //         }
  //       };

  //       await uploadData(exercisesData);

  //       const searchedExercises = exercisesData.filter(
  //         (item) =>
  //           item.name.toLowerCase().includes(search) ||
  //           item.target.toLowerCase().includes(search) ||
  //           item.equipment.toLowerCase().includes(search) ||
  //           item.bodyPart.toLowerCase().includes(search)
  //       );

  //       window.scrollTo({ top: 1800, left: 100, behavior: "smooth" });

  //       setSearch("");
  //       setExercises(searchedExercises);
  //     } catch (error) {
  //       console.error("Error searching for exercises:", error);
  //     }
  //   }
  // };

  return (
    <Stack alignItems="center" mt="37px" justifyContent="center" p="20px">
      <Typography
        fontWeight={700}
        sx={{ fontSize: { lg: "44px", xs: "30px" } }}
        mb="49px"
        textAlign="center"
        color="white"
      >
        Search exercises <br /> you wish to do
      </Typography>
      <Box position="relative" mb="72px">
        <TextField
          height="76px"
          sx={{
            input: { fontWeight: "700", border: "none", borderRadius: "4px" },
            width: { lg: "1170px", xs: "350px" },
            backgroundColor: "#fff",
            borderRadius: "4px",
          }}
          value={search}
          onChange={(e) => setSearch(e.target.value.toLowerCase())}
          placeholder="Search Exercises"
          type="text"
        />
        <Button
          className="search-btn"
          sx={{
            bgcolor: "#FF2625",
            color: "#fff",
            textTransform: "none",
            width: { lg: "173px", xs: "80px" },
            height: "56px",
            position: "absolute",
            right: "0px",
            fontSize: { lg: "20px", xs: "14px" },
          }}
          onClick={handleSearch}
        >
          Search
        </Button>
      </Box>
      <Box sx={{ position: "relative", width: "100%", p: "20px" }}>
        <HorizontalScrollbar
          data={bodyParts}
          bodyParts
          setBodyPart={setBodyPart}
          isBodyParts
          bodyPart={bodyPart}
        />
      </Box>
    </Stack>
  );
};

export default SearchExercises;
