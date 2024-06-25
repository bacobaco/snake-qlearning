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
let displayGame = false;

// Ajustement des hyperparamètres
let alpha = 0.1;  // Taux d'apprentissage réduit pour une convergence plus stable
let gamma = 0.99;  // Facteur de discount augmenté pour donner plus d'importance aux récompenses futures
let epsilon = 1.0;  // Taux d'exploration initial à 100%
const epsilonDecayRate = 0.9995;  // Taux de décroissance d'epsilon ralenti
const minEpsilon = 0.01;  // Valeur minimale d'epsilon inchangée

const qTable = {};  // Table Q

function getDistance(pos1, pos2) {
    return Math.sqrt(Math.pow(pos1.x - pos2.x, 2) + Math.pow(pos1.y - pos2.y, 2));
}

function calculateReward(oldHead, newHead, food, alive) {
    if (!alive) return -100;  // Pénalité réduite pour encourager l'exploration
    if (newHead.x === food.x && newHead.y === food.y) return 500;  // Récompense augmentée pour manger
    const oldDistance = getDistance(oldHead, food);
    const newDistance = getDistance(newHead, food);
    const distanceReward = (oldDistance - newDistance) * 20;  // Augmentation de l'importance de se rapprocher
    return distanceReward;
}

function getState() {
    const head = snake[0];
    const foodDirection = [
        Math.sign(food.x - head.x),  // -1 si la nourriture est à gauche, 1 si à droite, 0 si même colonne
        Math.sign(food.y - head.y)   // -1 si la nourriture est en haut, 1 si en bas, 0 si même ligne
    ];
    
    const directionState =
        direction === 'left' ? 0 :
        direction === 'right' ? 1 :
        direction === 'up' ? 2 :
        3;  // direction === 'down'
    
    const dangerAhead = isBlocked(
        head.x + (direction === 'right' ? 1 : direction === 'left' ? -1 : 0),
        head.y + (direction === 'down' ? 1 : direction === 'up' ? -1 : 0)
    );
    const dangerRight = isBlocked(
        head.x + (direction === 'down' ? 1 : direction === 'up' ? -1 : 0),
        head.y + (direction === 'left' ? 1 : direction === 'right' ? -1 : 0)
    );
    const dangerLeft = isBlocked(
        head.x + (direction === 'up' ? 1 : direction === 'down' ? -1 : 0),
        head.y + (direction === 'right' ? 1 : direction === 'left' ? -1 : 0)
    );
    
    return [...foodDirection, directionState, dangerAhead, dangerRight, dangerLeft];
}
function isBlocked(x, y) {
    return (
        x < 0 || x >= width / blockSize ||
        y < 0 || y >= height / blockSize ||
        snake.some(block => block.x === x && block.y === y)
    ) ? 1 : 0;
}

function chooseAction(state) {
    const stateKey = state.join('_');
    if (Math.random() < epsilon) {
        return Math.floor(Math.random() * 4);  // Exploration
    } else {
        if (!(stateKey in qTable)) {
            qTable[stateKey] = [0, 0, 0, 0];  // Initialiser si l'état n'existe pas
        }
        return qTable[stateKey].indexOf(Math.max(...qTable[stateKey]));  // Exploitation
    }
}

function updateQTable(state, action, reward, nextState) {
    const stateKey = state.join('_');
    const nextStateKey = nextState.join('_');
    if (!(stateKey in qTable)) qTable[stateKey] = [0, 0, 0, 0];
    if (!(nextStateKey in qTable)) qTable[nextStateKey] = [0, 0, 0, 0];
    const maxQ = Math.max(...qTable[nextStateKey]);
    qTable[stateKey][action] += alpha * (reward + gamma * maxQ - qTable[stateKey][action]);
}

function getRandomFood() {
    let newFood;
    do {
        newFood = {
            x: Math.floor(Math.random() * (width / blockSize)),
            y: Math.floor(Math.random() * (height / blockSize))
        };
    } while (snake.some(segment => segment.x === newFood.x && segment.y === newFood.y));
    return newFood;
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
    ctx.fillText(`Epsilon: ${epsilon.toFixed(4)}`, 10, 120);
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

function init() {
    snake = [{ x: 10, y: 10 }];
    food = getRandomFood();
    direction = 'right';
    score = 0;
    gamesPlayed++;
    if (gamesPlayed % 100 === 0) {
        console.log(`High Score after ${gamesPlayed} games: ${highScore}`);
    }
}

function updateEpsilon() {
    epsilon = Math.max(epsilon * epsilonDecayRate, minEpsilon);
}

async function gameLoop() {
    const oldHead = Object.assign({}, snake[0]);
    const state = getState();
    const action = chooseAction(state);
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
    const nextState = getState();
    const reward = calculateReward(oldHead, newHead, food, alive);
    updateQTable(state, action, reward, nextState);

    // Mise à jour d'epsilon après chaque action
    updateEpsilon();

    if (!alive) {
        init();
    }

    setTimeout(gameLoop, displayGame ? 50 : 0);
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