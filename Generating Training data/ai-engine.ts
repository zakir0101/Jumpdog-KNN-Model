import produce from "immer";
import _ from "lodash";

export type Piece = {
  id: string;
  color: "white" | "black";
  dragged?: boolean;
  isCaptured?: boolean;
  movePath?: string;
};
export type Square = number[];
export type Color = "white" | "black" | undefined;

export interface Board {
  whiteToPlay: boolean;
  isAnimating: boolean;
  pieces: (Piece | "")[][];
  moveLog: Move[];
  validMoves: Move[];
  hint: Move | null;
  hoverSquare: Square;
  playerClicks: Square[];
  isFirstClick: boolean;
}
export interface GameControls {
  hasGameStarted: boolean;
  isGameover: boolean;
  hasDismissedModal: boolean;
  level: number;
  color: Color;
}

export interface Store {
  board: Board;
  gameControls: GameControls;
}

export type Move = {
  id?: string;
  from: Square;
  to: Square;
  pieceMoved: Piece;
  path?: string;
  captures?: { [prop: string]: Piece };
};

export const getValidMoves = ({
  whiteToPlay,
  pieces,
}: {
  whiteToPlay: boolean;
  pieces: (Piece | "")[][];
}) => {
  const board = _.cloneDeep(pieces);
  board.forEach((row, x) => {
    row.forEach((piece, y) => {
      if (piece === "") return;
      if (piece.isCaptured) board[x][y] = "";
    });
  });
  const playerColor = whiteToPlay ? "white" : "black";
  const validMoves: Move[] = [];
  for (let row = 0; row < board.length; row++) {
    for (let col = 0; col < board[row].length; col++) {
      const piece = board[row][col];
      if (piece && piece.color == playerColor) {
        if (piece.isCaptured) continue;
        getDirectMoves(validMoves, row, col, piece, board);
        getCaptureMoves(validMoves, row, col, piece, board, playerColor);
      }
    }
  }
  return validMoves;
};

const getDirectMoves = (
  validMoves: Move[],
  pieceRow: number,
  pieceCol: number,
  piece: Piece,
  pieces: (Piece | "")[][]
) => {
  let directions = [
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 0],
  ];
  for (let [r, c] of directions) {
    const toRow = r + pieceRow;
    const toCol = c + pieceCol;
    if (4 >= toRow && toRow >= 0 && 4 >= toCol && toCol >= 0) {
      if (pieces[toRow][toCol] === "") {
        const from = [pieceRow, pieceCol];
        const to = [toRow, toCol];
        const pieceMoved: Piece = { ...piece };
        const move = createMove({
          from,
          to,
          pieceMoved,
          path: `${to[0]}${to[1]}`,
        });
        validMoves.push(move);
      }
    }
  }
};

const getCaptureMoves = (
  validMoves: Move[],
  pieceRow: number,
  pieceCol: number,
  piece: Piece,
  pieces: (Piece | "")[][],
  playerColor: string
) => {
  const opponentColor = playerColor === "white" ? "black" : "white";
  const pieceMoved = { ...piece };
  const from = [pieceRow, pieceCol];
  pieces = produce(pieces, (draft) => {
    draft[pieceRow][pieceCol] = "";
  });

  const squareCaptures = (
    validMoves: any,
    row: number,
    col: number,
    path: string = "",
    captures: any = {},
    visited: { [prop: string]: boolean } = {},
    exclude?: number[]
  ) => {
    if (visited[`${row}${col}`]) return;
    visited[`${row}${col}`] = true;

    let directions = [
      [-1, 0],
      [0, 1],
      [0, -1],
      [1, 0],
    ];
    if (exclude !== undefined) {
      directions = directions.filter(([r, c]) => {
        if (r === exclude[0] && c === exclude[1]) return false;
        return true;
      });
    }
    directions.forEach(([r, c]) => {
      const opponentRow = r + row;
      const opponentCol = c + col;
      const captureRow = r + r + row;
      const captureCol = c + c + col;
      if (
        4 >= captureRow &&
        captureRow >= 0 &&
        4 >= captureCol &&
        captureCol >= 0
      ) {
        const pieceCaptured = pieces[opponentRow][opponentCol];
        if (pieceCaptured && pieceCaptured.color === opponentColor) {
          if (pieces[captureRow][captureCol] === "") {
            const to = [captureRow, captureCol];
            const movePath: string =
              `${path} ${captureRow}${captureCol}`.trim();
            const moveCaptures = {
              ...captures,
              [String(opponentRow) + String(opponentCol)]: {
                ...pieceCaptured,
                isCaptured: true,
              },
            };
            const move = createMove({
              from,
              to,
              pieceMoved,
              path: movePath,
              captures: moveCaptures,
            });
            validMoves.push(move);
            const exclude = [-1 * r, -1 * c];
            const visitedSquares = { ...visited };
            squareCaptures(
              validMoves,
              captureRow,
              captureCol,
              movePath,
              moveCaptures,
              visitedSquares,
              exclude
            );
          }
        }
      }
    });
  };

  squareCaptures(validMoves, pieceRow, pieceCol);
};

export const createMove = ({
  from,
  to,
  pieceMoved,
  path = ``,
  captures = {},
}: Move) => {
  return {
    id:
      pieceMoved.id +
      (
        (from[0] + 1) * 100 * (to[1] + 1) * 100 +
        (to[0] + 1) * 100 * (from[1] + 1)
      ).toString(),
    from: from,
    to: to,
    path: path,
    pieceMoved: { ...pieceMoved, movePath: path },
    captures: captures,
  };
};


export const makeMove = (
  state: Store,
  move: Move,
) => {
  const {
    board: { pieces },
  } = state;
  const { from, to, pieceMoved, path } = move;
  const [toRow, toCol] = to;
  pieces[from[0]][from[1]] = "";
  pieces[toRow][toCol] = { ...pieceMoved, movePath: path };
  Object.entries(move.captures!).forEach(([position]) => {
    const [r, c] = [Number(position[0]), Number(position[1])];
    pieces[r][c] = "";
  });
};

export const undoMove = (state: Store, move: Move) => {
  const {
    board: { pieces },
  } = state;
  const { from, to, pieceMoved } = move!;
  const [toRow, toCol] = to;
  const reversdPath = move
    .path!.split(" ")
    .reverse()
    .slice(1)
    .concat([`${from[0]}${from[1]}`]);
  const movePath = reversdPath.join(" ");
  pieces[from[0]][from[1]] = { ...pieceMoved, movePath };
  pieces[toRow][toCol] = "";
  Object.entries(move.captures!).forEach(([position, pieceCaptured]) => {
    const [r, c] = [Number(position[0]), Number(position[1])];
    pieces[r][c] = { ...pieceCaptured, isCaptured: false };
  });
};

const GAME_OVER = 100;
const PIECE_WEIGHT = 5;

export const findBestMove = (state: Store) => {
  let {
    gameControls: { level },
    board: { whiteToPlay },
  } = state;
  let bestMove: Move | null = null;
  let DEPTH: number;

  level = Math.floor(Math.random() * 4) + 1
  switch (level) {
    case 1:
      DEPTH = 1;
      break;
    case 2:
      DEPTH = 4;
      break;
    case 3:
      DEPTH = 6;
      break;
    default:
      DEPTH = -1;
  }

  if (DEPTH == -1) {
    const validMoves = state.board.validMoves
    const moveIndex = Math.floor(Math.random() * validMoves.length)
    const move = validMoves[moveIndex]
    return move
  }

  const minMax = (
    state: Store,
    depth: number,
    alpha: number,
    beta: number,
    turnMultiplier: number
  ) => {
    let maxScore = -GAME_OVER / 100;
    state.board.validMoves = _.shuffle(state.board.validMoves);

    if (depth === 0) {
      return turnMultiplier * evaluateBoard(state);
    }
    for (let move of state.board.validMoves) {
      makeMove(state, move);
      state.board.whiteToPlay = !state.board.whiteToPlay;
      state.board.validMoves = getValidMoves({
        whiteToPlay: state.board.whiteToPlay,
        pieces: state.board.pieces,
      });

      const score = -minMax(state, depth - 1, -beta, -alpha, -turnMultiplier);
      if (score > maxScore) {
        maxScore = score;
        if (depth === DEPTH) {
          bestMove = move;
        }
      }
      undoMove(state, move);
      state.board.whiteToPlay = !state.board.whiteToPlay;
      state.board.validMoves = getValidMoves({
        whiteToPlay: state.board.whiteToPlay,
        pieces: state.board.pieces,
      });
      if (maxScore > alpha) {
        alpha = maxScore;
      }
      if (alpha >= beta) {
        break;
      }
    }

    return maxScore;
  };

  const score = minMax(state, DEPTH, -GAME_OVER, GAME_OVER, state.board.whiteToPlay ? 1 : -1);

  if (!bestMove) {
    bestMove = _.shuffle(state.board.validMoves)[0];
  }
  return bestMove;
};

export const findBestMoveScore = (state: Store) => {
  let {
    gameControls: { level },
    board: { whiteToPlay },
  } = state;
  let bestMove: Move | null = null;
  let DEPTH: number = 6;


  const minMax = (
    state: Store,
    depth: number,
    alpha: number,
    beta: number,
    turnMultiplier: number
  ) => {
    let maxScore = -GAME_OVER / 100;
    state.board.validMoves = _.shuffle(state.board.validMoves);

    if (depth === 0) {
      return turnMultiplier * evaluateBoard(state);
    }
    for (let move of state.board.validMoves) {
      makeMove(state, move);
      state.board.whiteToPlay = !state.board.whiteToPlay;
      state.board.validMoves = getValidMoves({
        whiteToPlay: state.board.whiteToPlay,
        pieces: state.board.pieces,
      });

      const score = -minMax(state, depth - 1, -beta, -alpha, -turnMultiplier);
      if (score > maxScore) {
        maxScore = score;
        if (depth === DEPTH) {
          bestMove = move;
        }
      }
      undoMove(state, move);
      state.board.whiteToPlay = !state.board.whiteToPlay;
      state.board.validMoves = getValidMoves({
        whiteToPlay: state.board.whiteToPlay,
        pieces: state.board.pieces,
      });
      if (maxScore > alpha) {
        alpha = maxScore;
      }
      if (alpha >= beta) {
        break;
      }
    }

    return maxScore;
  };

  const score = minMax(state, DEPTH, -GAME_OVER, GAME_OVER, state.board.whiteToPlay ? 1 : -1);

  if (!bestMove) {
    bestMove = _.shuffle(state.board.validMoves)[0];
  }
  return score;
};

const evaluateBoard = (state: Store) => {
  const {
    board: { pieces },
  } = state;
  let score = 0;
  let whiteCount = 0;
  let blackCount = 0;
  for (let row of pieces) {
    for (let piece of row) {
      if (piece === "") continue;
      if (piece.color === "white") {
        score += PIECE_WEIGHT;
        whiteCount += 1;
      }
      if (piece.color === "black") {
        score -= PIECE_WEIGHT;
        blackCount += 1;
      }
    }
  }
  if (whiteCount === 0) {
    score = GAME_OVER;
  } else if (blackCount === 0) {
    score = GAME_OVER;
  }
  return score / 100;
};
