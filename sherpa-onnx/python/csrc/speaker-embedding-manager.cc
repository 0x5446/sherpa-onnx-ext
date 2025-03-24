// sherpa-onnx/python/csrc/speaker-embedding-manager.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/speaker-embedding-manager.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/speaker-embedding-manager.h"

namespace sherpa_onnx {

/// modified by tf @2025-03-24
/// add VerificationResult and SpeakerMatch
/// add verify_with_score and search_with_score
void PybindSpeakerEmbeddingManager(py::module *m) {
  using PyClass = SpeakerEmbeddingManager;

  py::class_<VerificationResult>(*m, "VerificationResult")
      .def_readonly("match", &VerificationResult::match)
      .def_readonly("score", &VerificationResult::score);

  py::class_<SpeakerMatch>(*m, "SpeakerMatch")
      .def_readonly("name", &SpeakerMatch::name)
      .def_readonly("score", &SpeakerMatch::score);

  py::class_<PyClass>(*m, "SpeakerEmbeddingManager")
      .def(py::init<int32_t>(), py::arg("dim"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("num_speakers", &PyClass::NumSpeakers)
      .def_property_readonly("dim", &PyClass::Dim)
      .def_property_readonly("all_speakers", &PyClass::GetAllSpeakers)
      .def(
          "__contains__",
          [](const PyClass &self, const std::string &name) -> bool {
            return self.Contains(name);
          },
          py::arg("name"), py::call_guard<py::gil_scoped_release>())
      .def(
          "add",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v) -> bool {
            return self.Add(name, v.data());
          },
          py::arg("name"), py::arg("v"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add",
          [](const PyClass &self, const std::string &name,
             const std::vector<std::vector<float>> &embedding_list) -> bool {
            return self.Add(name, embedding_list);
          },
          py::arg("name"), py::arg("embedding_list"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "remove",
          [](const PyClass &self, const std::string &name) -> bool {
            return self.Remove(name);
          },
          py::arg("name"), py::call_guard<py::gil_scoped_release>())
      .def(
          "search",
          [](const PyClass &self, const std::vector<float> &v, float threshold)
              -> std::string { return self.Search(v.data(), threshold); },
          py::arg("v"), py::arg("threshold"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "search_with_score",
          [](const PyClass &self, const std::vector<float> &v, float threshold)
              -> SpeakerMatch { return self.SearchWithScore(v.data(), threshold); },
          py::arg("v"), py::arg("threshold"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_best_matches",
          [](const PyClass &self, const std::vector<float> &v, float threshold,
             int32_t n) -> std::vector<SpeakerMatch> {
            return self.GetBestMatches(v.data(), threshold, n);
          },
          py::arg("v"), py::arg("threshold"), py::arg("n"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "verify",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v, float threshold) -> bool {
            return self.Verify(name, v.data(), threshold);
          },
          py::arg("name"), py::arg("v"), py::arg("threshold"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "verify_with_score",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v, float threshold) -> VerificationResult {
            return self.VerifyWithScore(name, v.data(), threshold);
          },
          py::arg("name"), py::arg("v"), py::arg("threshold"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "score",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v) -> float {
            return self.Score(name, v.data());
          },
          py::arg("name"), py::arg("v"),
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
