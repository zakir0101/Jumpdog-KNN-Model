const fs = require('fs');
const path = require('path');
const crypto = require("crypto");
import _ from "lodash";
import { Store, Piece, Color, findBestMove, getValidMoves, makeMove, Move, Board, undoMove, findBestMoveScore } from "./ai-engine";


export const initializePieces = (color: Color): (Piece | "")[][] => {
    const pieces: (Piece | "")[][] = [];
    for (let r = 0; r <= 4; r++) {
        const row: ("" | Piece)[] = [];
        for (let c = 0; c <= 4; c++) {
            if (r <= 1)
                row.push({
                    id: String(Math.random() * 1000),
                    color: color === "white" ? "black" : "white",
                });
            if (r >= 3)
                row.push({
                    id: String(Math.random() * 1000),
                    color: color === "white" ? "white" : "black",
                });
            if (r === 2 && c === 2) row.push("");
            if (r === 2 && c >= 3)
                row.push({
                    id: String(Math.random() * 1000),
                    color: color === "white" ? "black" : "white",
                });
            if (r === 2 && c <= 1)
                row.push({
                    id: String(Math.random() * 1000),
                    color: color === "white" ? "white" : "black",
                });
        }
        pieces.push(row);
    }
    return pieces;
};


function initialState(): Store {
    return {
        gameControls: {
            hasGameStarted: false,
            isGameover: false,
            hasDismissedModal: false,
            color: undefined,
            level: 3,
        },
        board: {
            hint: null,
            isAnimating: false,
            moveLog: [],
            validMoves: [],
            whiteToPlay: true,
            hoverSquare: [],
            pieces: initializePieces("white"),
            playerClicks: [],
            isFirstClick: true,
        },
    }
}

function makePlayerMove(state: Store, move: Move) {
    const board = state.board
    makeMove(state, move);
    board.whiteToPlay = !state.board.whiteToPlay;
    board.moveLog.push(move);
    board.validMoves = getValidMoves({
        whiteToPlay: board.whiteToPlay,
        pieces: board.pieces,
    });
    state.gameControls.isGameover = board.validMoves.length === 0;
    return state;
}


function parseBoard(board: Board) {
    const boardStruct: number[][] = []
    for (const row of board.pieces) {
        const rowStruct = row.map(p => {
            if (p == '') return 0
            if (p.color == 'black') return -1
            else return 1

        })
        boardStruct.push(rowStruct)
    }
    return boardStruct
}

function evaluatePositions(state: Store) {
    const evaluations: object[] = []
    const board = state.board
    for (const move of board.validMoves) {
        makeMove(state, move);
        state.board.whiteToPlay = !state.board.whiteToPlay;
        state.board.validMoves = getValidMoves({
            whiteToPlay: state.board.whiteToPlay,
            pieces: state.board.pieces,
        });
        const score = findBestMoveScore(_.cloneDeep(state));
        evaluations.push({ board: parseBoard(state.board), score })
        undoMove(state, move);
        state.board.whiteToPlay = !state.board.whiteToPlay;
        state.board.validMoves = getValidMoves({
            whiteToPlay: state.board.whiteToPlay,
            pieces: state.board.pieces,
        });
    }
    return evaluations
}

function playGame() {
    let uuid = crypto.randomUUID();
    const filePath = path.join(__dirname, `${uuid}.json`);

    const state = initialState()
    const board = state.board
    board.validMoves = getValidMoves({
        whiteToPlay: board.whiteToPlay,
        pieces: board.pieces,
    });

    let gamePositions: object[] = []
    while (!state.gameControls.isGameover) {
        const evaluations: object[] = evaluatePositions(_.cloneDeep(state))
        gamePositions = [...gamePositions, evaluations]
        const bestMove = findBestMove(_.cloneDeep(state));
        makePlayerMove(state, bestMove);
    }

    const jsonString = JSON.stringify(gamePositions, null, 2);
    try {
        fs.writeFileSync(filePath, jsonString);
        console.log('File successfully written:', filePath);
    } catch (err) {
        console.error('Error writing file:', err);
    }
}

for (let i = 0; i < 4; i++) {
    console.log("------------------");
    console.log("Started Game ", i);
    playGame()
    console.log("Finished Game ", i);
}