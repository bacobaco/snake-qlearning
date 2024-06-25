// Initialisation du canvas
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const width = canvas.width = 400;
const height = canvas.height = 400;
const blockSize = 20;

// Variables globales
let snake, food, direction, score;
let highScore = 0;
let gamesPlayed = 0;
let nb_moves = 0;
let displayGame = false;
const scores = [];

// Création du réseau neuronal PPO
const inputSize = (width / blockSize) * (height / blockSize) + 4;
const hiddenSize = 64;
const outputSize = 4;

function getDistance(pos1, pos2) {
    return Math.sqrt(Math.pow(pos1.x - pos2.x, 2) + Math.pow(pos1.y - pos2.y, 2));
}

function calculateReward(oldHead, newHead, food, alive) {
    if (!alive) return -1000;
    if (newHead.x === food.x && newHead.y === food.y) return 100;
    const oldDistance = getDistance(oldHead, food);
    const newDistance = getDistance(newHead, food);
    const distanceReward = (oldDistance - newDistance)*10 ;
    const movePenalty = nb_moves > 100 ? -0.1 * (nb_moves - 100) : 0;
    return distanceReward ;//+ movePenalty;
}

class PPO {
    constructor(inputSize, hiddenSize, outputSize) {
        this.actor = tf.sequential();
        this.actor.add(tf.layers.dense({ units: hiddenSize, inputShape: [inputSize], activation: 'relu' }));
        this.actor.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
        this.actor.add(tf.layers.dense({ units: outputSize, activation: 'softmax' }));
        this.actor.compile({ optimizer: tf.train.adam(0.0003), loss: 'categoricalCrossentropy' });

        this.critic = tf.sequential();
        this.critic.add(tf.layers.dense({ units: hiddenSize, inputShape: [inputSize], activation: 'relu' }));
        this.critic.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
        this.critic.add(tf.layers.dense({ units: 1, activation: 'linear' }));
        this.critic.compile({ optimizer: tf.train.adam(0.0003), loss: 'meanSquaredError' });

        this.gamma = 0.99;
        this.epsilon = 0.2;
        this.memory = [];
        this.batchSize = 64;
    }

    getState(snake, food) {
        const grid = Array.from({ length: height / blockSize }, () => Array(width / blockSize).fill(0));
        const head = snake[0];
        grid[head.y][head.x] = 1;
        snake.slice(1).forEach(block => grid[block.y][block.x] = 2);
        grid[food.y][food.x] = 3;
        const state = grid.flat();
        const directionState = [
            direction === 'left' ? 1 : 0,
            direction === 'right' ? 1 : 0,
            direction === 'up' ? 1 : 0,
            direction === 'down' ? 1 : 0
        ];
        return state.concat(directionState);
    }

    act(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const probabilities = this.actor.predict(stateTensor);
            const action = tf.multinomial(probabilities, 1).dataSync()[0];
            return action;
        });
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push({ state, action, reward, nextState, done });
    }

    async train() {
        if (this.memory.length < this.batchSize) return;

        const batch = this.memory.slice(-this.batchSize);
        this.memory = [];

        const states = tf.tensor2d(batch.map(exp => exp.state));
        const actions = tf.tensor1d(batch.map(exp => exp.action), 'int32');
        const rewards = tf.tensor1d(batch.map(exp => exp.reward));
        const nextStates = tf.tensor2d(batch.map(exp => exp.nextState));
        const dones = tf.tensor1d(batch.map(exp => exp.done ? 0 : 1));

        const values = this.critic.predict(states);
        const nextValues = this.critic.predict(nextStates);

        const returns = tf.tidy(() => {
            const discountedRewards = rewards.add(nextValues.squeeze().mul(dones).mul(this.gamma));
            return discountedRewards.sub(values.squeeze());
        });

        const oldProbabilities = this.actor.predict(states);

        for (let epoch = 0; epoch < 5; epoch++) {
            // Actor training
            await this.actor.optimizer.minimize(() => {
                return tf.tidy(() => {
                    const newProbabilities = this.actor.predict(states);
                    const actionsOneHot = tf.oneHot(actions, 4);
                    const newProbabilitiesActions = newProbabilities.mul(actionsOneHot).sum(1);
                    const oldProbabilitiesActions = oldProbabilities.mul(actionsOneHot).sum(1);

                    const ratios = newProbabilitiesActions.div(oldProbabilitiesActions);
                    const clippedRatios = tf.clipByValue(ratios, 1 - this.epsilon, 1 + this.epsilon);

                    const surrogate1 = ratios.mul(returns);
                    const surrogate2 = clippedRatios.mul(returns);

                    return tf.mean(tf.minimum(surrogate1, surrogate2)).neg();
                });
            });

            // Critic training
            await this.critic.optimizer.minimize(() => {
                return tf.tidy(() => {
                    const targets = rewards.add(nextValues.squeeze().mul(dones).mul(this.gamma));
                    const predictions = this.critic.predict(states).squeeze();
                    return tf.losses.meanSquaredError(targets, predictions);
                });
            });
        }

        tf.dispose([states, actions, rewards, nextStates, dones, values, nextValues, returns, oldProbabilities]);
    }
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
    ctx.fillText(`Snake moves: ${nb_moves}`, 10, 120);
}

function logScore(score) {
    scores.push(score);
    console.log(`Games Played: ${gamesPlayed}, Score: ${score}`);
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
        nb_moves = 0;
        if (score > highScore) highScore = score;
        food = getRandomFood();
    } else {
        nb_moves++;
        snake.pop();
    }
    return true;
}

const ppo = new PPO(inputSize, hiddenSize, outputSize);

async function gameLoop() {
    const oldHead = Object.assign({}, snake[0]);
    const state = ppo.getState(snake, food);
    const action = ppo.act(state);
    switch (action) {
        case 0: if (direction !== 'right') direction = 'left'; break;
        case 1: if (direction !== 'left') direction = 'right'; break;
        case 2: if (direction !== 'down') direction = 'up'; break;
        case 3: if (direction !== 'up') direction = 'down'; break;
    }

    const alive = move();
    if (displayGame) {
        draw();
    }

    const newHead = snake[0];
    const nextState = ppo.getState(snake, food);
    const reward = calculateReward(oldHead, newHead, food, alive);
    ppo.remember(state, action, reward, nextState, !alive);

    if (!alive) {
        await ppo.train(); 
        init();
    }

    setTimeout(gameLoop, displayGame? 50 :0);
}

function init() {
    snake = [{ x: 10, y: 10 }];
    food = getRandomFood();
    direction = 'right';
    score = 0;
    nb_moves = 0;
    gamesPlayed++;
    if (gamesPlayed % 100 === 0) {
        console.log(`High Score after ${gamesPlayed} games: ${highScore}`);
    }
}

document.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
        displayGame = !displayGame;
        if (displayGame) {
            draw();
        }
    }
});

init();
gameLoop();