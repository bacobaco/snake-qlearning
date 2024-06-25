// Initialisation du canvas
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const width = canvas.width = 400;
const height = canvas.height = 400;
const blockSize = 20;

// Variables globales
let snake, food, direction, game, score;
let highScore = 0;
let gamesPlayed = 0;

// Création du réseau neuronal DQN
const inputSize = 7;
const hiddenSize = 16;
const outputSize = 4;

class DQN {
    constructor(inputSize, hiddenSize, outputSize) {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: hiddenSize, inputShape: [inputSize], activation: 'relu' }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));
        this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        
        this.epsilon = 1;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.gamma = 0.95;
        this.memory = [];
        this.batchSize = 32;
    }

    getState(snake, food) {
        const head = snake[0];
        const state = [
            (direction === 'left') ? 1 : 0,
            (direction === 'right') ? 1 : 0,
            (direction === 'up') ? 1 : 0,
            (direction === 'down') ? 1 : 0,
            food.x < head.x ? 1 : 0,
            food.y < head.y ? 1 : 0,
            snake.length
        ];
        return state;
    }

    act(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 4);
        }
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const prediction = this.model.predict(stateTensor);
            return prediction.argMax(1).dataSync()[0];
        });
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push([state, action, reward, nextState, done]);
    }

    replay() {
        if (this.memory.length < this.batchSize) return;

        const batch = this.memory.slice(-this.batchSize);
        const states = batch.map(exp => exp[0]);
        const nextStates = batch.map(exp => exp[3]);

        const predictionsQ = tf.tidy(() => this.model.predict(tf.tensor2d(states)));
        const predictionsNextQ = tf.tidy(() => this.model.predict(tf.tensor2d(nextStates)));

        const x = [];
        const y = [];

        for (let i = 0; i < this.batchSize; i++) {
            const [state, action, reward, nextState, done] = batch[i];
            const currentQ = predictionsQ.dataSync().slice(i * 4, (i + 1) * 4);
            const nextQ = predictionsNextQ.dataSync().slice(i * 4, (i + 1) * 4);

            if (done) {
                currentQ[action] = reward;
            } else {
                currentQ[action] = reward + this.gamma * Math.max(...nextQ);
            }

            x.push(state);
            y.push(currentQ);
        }

        this.model.fit(tf.tensor2d(x), tf.tensor2d(y), { epochs: 1, verbose: 0 });

        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }

        predictionsQ.dispose();
        predictionsNextQ.dispose();
    }
}

const dqn = new DQN(inputSize, hiddenSize, outputSize);

// Fonctions du jeu
function init() {
    snake = [{ x: 10, y: 10 }];
    food = getRandomFood();
    direction = 'right';
    score = 0;
    gamesPlayed++;
}

function getRandomFood() {
    return {
        x: Math.floor(Math.random() * (width / blockSize)),
        y: Math.floor(Math.random() * (height / blockSize))
    };
}

function draw() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = 'lime';
    for (let block of snake) {
        ctx.fillRect(block.x * blockSize, block.y * blockSize, blockSize, blockSize);
    }

    ctx.fillStyle = 'red';
    ctx.fillRect(food.x * blockSize, food.y * blockSize, blockSize, blockSize);

    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.fillText(`Score: ${score}`, 10, 30);
    ctx.fillText(`High Score: ${highScore}`, 10, 60);
    ctx.fillText(`Games Played: ${gamesPlayed}`, 10, 90);
}

function move() {
    const head = Object.assign({}, snake[0]);

    switch (direction) {
        case 'left': head.x--; break;
        case 'right': head.x++; break;
        case 'up': head.y--; break;
        case 'down': head.y++; break;
    }

    if (head.x < 0 || head.x >= width / blockSize ||
        head.y < 0 || head.y >= height / blockSize ||
        snake.some(block => block.x === head.x && block.y === head.y)) {
        return false;
    }

    snake.unshift(head);

    if (head.x === food.x && head.y === food.y) {
        score++;
        if (score > highScore) highScore = score;
        food = getRandomFood();
    } else {
        snake.pop();
    }

    return true;
}

function gameLoop() {
    const state = dqn.getState(snake, food);
    const action = dqn.act(state);

    switch (action) {
        case 0: if (direction !== 'right') direction = 'left'; break;
        case 1: if (direction !== 'left') direction = 'right'; break;
        case 2: if (direction !== 'down') direction = 'up'; break;
        case 3: if (direction !== 'up') direction = 'down'; break;
    }

    const alive = move();
    draw();

    const nextState = dqn.getState(snake, food);
    const reward = alive ? (score > 0 ? 1 : 0) : -1;

    dqn.remember(state, action, reward, nextState, !alive);
    dqn.replay();

    if (alive) {
        setTimeout(gameLoop, 0);
    } else {
        console.log(`Game Over. Score: ${score}`);
        init();
        setTimeout(gameLoop, 0);
    }
}

init();
gameLoop();