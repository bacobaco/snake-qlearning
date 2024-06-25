const GRID_SIZE = 20;
const CELL_SIZE = 20;
const INITIAL_SNAKE_LENGTH = 3;
const POPULATION_SIZE = 50;
const MUTATION_RATE = 0.1;

const INPUT_NODES = 11;
const HIDDEN_NODES = 16;
const OUTPUT_NODES = 4;

let canvas, ctx;
let food;
let score, generation;
let population, currentSnakeIndex;
let spacePressed = false;
let bestScore = { score: 0, generation: 0 };

document.addEventListener('keydown', (event) => {
  if (event.code === 'Space') {
    spacePressed = true;
  }
});

document.addEventListener('keyup', (event) => {
  if (event.code === 'Space') {
    spacePressed = false;
  }
});
function drawBackground() {
  ctx.globalAlpha = 0.5; // Set transparency

  // Draw checkerboard pattern
  for (let i = 0; i < GRID_SIZE; i++) {
    for (let j = 0; j < GRID_SIZE; j++) {
      ctx.fillStyle = (i + j) % 2 === 0 ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    }
  }

  // Draw grid lines
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= GRID_SIZE; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL_SIZE, 0);
    ctx.lineTo(i * CELL_SIZE, GRID_SIZE * CELL_SIZE);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i * CELL_SIZE);
    ctx.lineTo(GRID_SIZE * CELL_SIZE, i * CELL_SIZE);
    ctx.stroke();
  }

  ctx.globalAlpha = 1; // Reset transparency
}

class NeuralNetwork {
  constructor(inputs, hidden, outputs) {
    this.inputNodes = inputs;
    this.hiddenNodes = hidden;
    this.outputNodes = outputs;

    this.weightsIH = Matrix.randomize(this.hiddenNodes, this.inputNodes);
    this.weightsHO = Matrix.randomize(this.outputNodes, this.hiddenNodes);
    this.biasH = Matrix.randomize(this.hiddenNodes, 1);
    this.biasO = Matrix.randomize(this.outputNodes, 1);
  }

  feedForward(inputArray) {
    let inputs = Matrix.fromArray(inputArray);
    let hidden = Matrix.multiply(this.weightsIH, inputs);
    hidden.add(this.biasH);
    hidden.map(sigmoid);

    let output = Matrix.multiply(this.weightsHO, hidden);
    output.add(this.biasO);
    output.map(sigmoid);

    return output.toArray();
  }

  mutate(rate) {
    function mutate(val) {
      if (Math.random() < rate) {
        return val + Math.random() * 0.5 - 0.25;
      }
      return val;
    }

    this.weightsIH.map(mutate);
    this.weightsHO.map(mutate);
    this.biasH.map(mutate);
    this.biasO.map(mutate);
  }

  static crossover(a, b) {
    let child = new NeuralNetwork(a.inputNodes, a.hiddenNodes, a.outputNodes);
    child.weightsIH = Matrix.crossover(a.weightsIH, b.weightsIH);
    child.weightsHO = Matrix.crossover(a.weightsHO, b.weightsHO);
    child.biasH = Matrix.crossover(a.biasH, b.biasH);
    child.biasO = Matrix.crossover(a.biasO, b.biasO);
    return child;
  }
}

class Snake {
  constructor(brain) {
    this.brain = brain || new NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);
    this.fitness = 0;
    this.reset();
  }

  reset() {
    this.body = [];
    for (let i = 0; i < INITIAL_SNAKE_LENGTH; i++) {
      this.body.push({ x: Math.floor(GRID_SIZE / 2) - i, y: Math.floor(GRID_SIZE / 2) });
    }
    this.direction = { x: 1, y: 0 };
    this.alive = true;
    this.lifetime = 0;
    this.foodEaten = 0;
    this.movesSinceLastFood = 0;
    this.movesSinceLastTurn = 0;
    this.lastDirection = { x: 1, y: 0 };
  }

  move() {
    if (!this.alive) return;

    let head = { x: this.body[0].x + this.direction.x, y: this.body[0].y + this.direction.y };

    let oldDistance = Math.abs(this.body[0].x - food.x) + Math.abs(this.body[0].y - food.y);
    let newDistance = Math.abs(head.x - food.x) + Math.abs(head.y - food.y);

    if (newDistance < oldDistance) {
      this.fitness += 5;
    } else if (newDistance > oldDistance) {
      this.fitness -= 5;
    }

    if (head.x < 0 || head.x >= GRID_SIZE || head.y < 0 || head.y >= GRID_SIZE) {
      this.alive = false;
      this.fitness -= 100;
      return;
    }

    for (let i = 1; i < this.body.length; i++) {
      if (head.x === this.body[i].x && head.y === this.body[i].y) {
        this.alive = false;
        this.fitness -= 100;
        return;
      }
    }

    this.body.unshift(head);

    if (head.x === food.x && head.y === food.y) {
      score++;
      this.foodEaten++;
      this.fitness += 10;
      this.movesSinceLastFood = 0;
      generateFood();
    } else {
      this.body.pop();
      this.movesSinceLastFood++;
    }

    this.lifetime++;
    //this.fitness += 1;

    if (this.movesSinceLastFood > 100) {
      this.fitness -= 5;
    }

    if (this.direction.x === this.lastDirection.x && this.direction.y === this.lastDirection.y) {
      this.movesSinceLastTurn++;
      if (this.movesSinceLastTurn > 10) {
        this.fitness -= 5;
      }
    } else {
      this.movesSinceLastTurn = 0;
      this.lastDirection = { ...this.direction };
    }
  }

  think() {
    let inputs = this.getInputs();
    let output = this.brain.feedForward(inputs);
    let maxIndex = output.indexOf(Math.max(...output));

    switch (maxIndex) {
      case 0:
        if (this.direction.y === 0) this.direction = { x: 0, y: -1 };
        break;
      case 1:
        if (this.direction.y === 0) this.direction = { x: 0, y: 1 };
        break;
      case 2:
        if (this.direction.x === 0) this.direction = { x: -1, y: 0 };
        break;
      case 3:
        if (this.direction.x === 0) this.direction = { x: 1, y: 0 };
        break;
    }
  }

  getInputs() {
    let head = this.body[0];
    let inputs = [];

    inputs.push(head.x / GRID_SIZE);
    inputs.push(head.y / GRID_SIZE);
    inputs.push((GRID_SIZE - 1 - head.x) / GRID_SIZE);
    inputs.push((GRID_SIZE - 1 - head.y) / GRID_SIZE);

    inputs.push(Math.sign(food.x - head.x));
    inputs.push(Math.sign(food.y - head.y));

    inputs.push(this.direction.x);
    inputs.push(this.direction.y);

    inputs.push(this.checkDanger(this.direction));

    let rightDir = { x: this.direction.y, y: -this.direction.x };
    inputs.push(this.checkDanger(rightDir));

    let leftDir = { x: -this.direction.y, y: this.direction.x };
    inputs.push(this.checkDanger(leftDir));

    return inputs;
  }

  checkDanger(dir) {
    let head = this.body[0];
    let next = { x: head.x + dir.x, y: head.y + dir.y };

    if (next.x < 0 || next.x >= GRID_SIZE || next.y < 0 || next.y >= GRID_SIZE) {
      return 1;
    }

    for (let i = 1; i < this.body.length; i++) {
      if (next.x === this.body[i].x && next.y === this.body[i].y) {
        return 1;
      }
    }

    return 0;
  }

  draw() {
    
  drawBackground()
    ctx.fillStyle = 'green';
    for (let part of this.body) {
      ctx.fillRect(part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    }

    ctx.fillStyle = 'white';
    let head = this.body[0];
    let eyeSize = CELL_SIZE / 4;
    let eyeOffset = CELL_SIZE / 4;
    ctx.fillRect(head.x * CELL_SIZE + eyeOffset, head.y * CELL_SIZE + eyeOffset, eyeSize, eyeSize);
    ctx.fillRect(head.x * CELL_SIZE + CELL_SIZE - eyeOffset - eyeSize, head.y * CELL_SIZE + eyeOffset, eyeSize, eyeSize);
  }
}

function generateFood() {
  let newFood;
  do {
    newFood = {
      x: Math.floor(Math.random() * GRID_SIZE),
      y: Math.floor(Math.random() * GRID_SIZE)
    };
  } while (population[currentSnakeIndex].body.some(part => part.x === newFood.x && part.y === newFood.y));
  food = newFood;
}

function drawFood() {
  ctx.fillStyle = 'red';
  ctx.beginPath();
  ctx.arc(
    (food.x + 0.5) * CELL_SIZE,
    (food.y + 0.5) * CELL_SIZE,
    CELL_SIZE / 2,
    0,
    2 * Math.PI
  );
  ctx.fill();
}

function initializePopulation() {
  population = [];
  for (let i = 0; i < POPULATION_SIZE; i++) {
    population.push(new Snake());
  }
  currentSnakeIndex = 0;
}

function nextGeneration() {
  calculateFitness();
  let newPopulation = [];

  let sortedPopulation = population.sort((a, b) => b.fitness - a.fitness);
  newPopulation.push(sortedPopulation[0]);
  newPopulation.push(sortedPopulation[1]);

  while (newPopulation.length < POPULATION_SIZE) {
    let parentA = selectParent();
    let parentB = selectParent();
    let child = new Snake(NeuralNetwork.crossover(parentA.brain, parentB.brain));
    child.brain.mutate(MUTATION_RATE);
    newPopulation.push(child);
  }
  population = newPopulation;
  currentSnakeIndex = 0;
  generation++;
  if (score > bestScore.score) {
    bestScore.score = score;
    bestScore.generation = generation - 1;
  }
  score = 0;
}

function calculateFitness() {
  let maxFitness = 0;
  for (let snake of population) {
    snake.fitness = Math.pow(snake.foodEaten, 2) * 1000 + snake.lifetime;
    maxFitness = Math.max(maxFitness, snake.fitness);
  }
  for (let snake of population) {
    snake.fitness /= maxFitness;
  }
}

function selectParent() {
  let r = Math.random();
  let index = 0;
  while (r > 0 && index < population.length) {
    r -= population[index].fitness;
    index++;
  }
  index--;
  return population[index];
}

function gameLoop() {
  let currentSnake = population[currentSnakeIndex];
  if (currentSnake.alive) {
    currentSnake.think();
    currentSnake.move();
  } else {
    currentSnakeIndex++;
    if (currentSnakeIndex >= POPULATION_SIZE) {
      nextGeneration();
    }
    currentSnake = population[currentSnakeIndex];
    currentSnake.reset();
    generateFood();
  }

  if (spacePressed) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    currentSnake.draw();
    drawFood();
    ctx.fillStyle = 'black';
    ctx.font = '20px Arial';
    ctx.fillText(`Score: ${score}`, 10, 30);
    ctx.fillText(`Best Score: ${bestScore.score} (Gen ${bestScore.generation})`, 10, 60);
    ctx.fillText(`Generation: ${generation}`, 10, 90);
    ctx.fillText(`Snake: ${currentSnakeIndex + 1}/${POPULATION_SIZE}`, 10, 120);
  }

  requestAnimationFrame(gameLoop);
}

function initGame() {
  canvas = document.createElement('canvas');
  canvas.width = GRID_SIZE * CELL_SIZE;
  canvas.height = GRID_SIZE * CELL_SIZE;
  document.body.appendChild(canvas);
  ctx = canvas.getContext('2d');

  generation = 1;
  score = 0;
  initializePopulation();
  generateFood();
  gameLoop();
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = [];

    for (let i = 0; i < this.rows; i++) {
      this.data[i] = [];
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = 0;
      }
    }
  }

  static randomize(rows, cols) {
    let result = new Matrix(rows, cols);
    result.map(() => Math.random() * 2 - 1);
    return result;
  }

  static multiply(a, b) {
    if (a.cols !== b.rows) {
      console.error('Columns of A must match rows of B.');
      return undefined;
    }

    let result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  static fromArray(arr) {
    let m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  static crossover(a, b) {
    let result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < a.cols; j++) {
        result.data[i][j] = Math.random() < 0.5 ? a.data[i][j] : b.data[i][j];
      }
    }
    return result;
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  add(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n;
        }
      }
    }
  }

  map(func) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        let val = this.data[i][j];
        this.data[i][j] = func(val);
      }
    }
  }
}

initGame();
