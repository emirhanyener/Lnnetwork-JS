function csv_to_json(value) {
  const lines = value.split("\n");
  const headers = lines[0].split(",");

  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const output = {};
    headers.forEach((header, index) => {
      output[header] = values[index];
    });
    return output;
  });
}
